"""
Tissue-specific splice site evaluation for SpliceMamba, Pangolin, and SpliceAI.

Evaluates all models using the Pangolin paper methodology:
  - Ground truth: ANNOTATION-BASED splice site labels (same for all tissues)
  - Predictions: tissue-specific model output for Pangolin; same predictions
    for tissue-agnostic models (SpliceMamba, SpliceAI)

This matches how the Pangolin paper evaluates: the tissue specificity is in
the MODEL predictions, not in the labels. Each Pangolin tissue model is scored
against the same annotation labels to show it maintains accuracy.

A separate tissue-differential section analyzes sites active in one tissue
but not another (using GTEx labels), measuring whether Pangolin's tissue
models score tissue-specific sites higher.

Usage:
    python evaluate_tissue.py --splicemamba-ckpt checkpoints/best.pt
    python evaluate_tissue.py --splicemamba-ckpt checkpoints/best.pt \
        --pangolin-preds results/pangolin_preds.npz \
        --spliceai-preds results/spliceai_preds.npz
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import h5py
import numpy as np
import torch

from model import SpliceMamba
from eval_utils import (
    compute_gene_window_counts,
    stitch_gene_predictions,
    read_window_labels,
    stitch_gene_labels,
    labels_to_binary,
    adapt_to_binary_splice,
    compute_binary_auprc,
    compute_binary_topk,
    compute_binary_f1,
    compute_binary_positional,
    make_serializable,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TISSUES = ["heart", "liver", "brain", "testis"]

CONFIG = dict(
    test_dataset_path=str(Path(_REPO_ROOT) / "dataset_test_0.h5"),
    test_datafile_path=str(Path(_REPO_ROOT) / "datafile_test_0.h5"),
    tissue_labels_path=str(Path(_REPO_ROOT) / "gtex_tissue_labels_chr1.h5"),
    d_model=256,
    n_mamba_layers=8,
    d_state=64,
    expand=2,
    d_conv=4,
    headdim=32,
    n_attn_layers=4,
    n_heads=8,
    window_radius=400,
    dropout=0.15,
    drop_path_rate=0.0,
    n_classes=3,
    max_len=15000,
    label_start=5000,
    label_end=10000,
    batch_size=4,
    peak_height=0.3,
    peak_distance=20,
)


# ---------------------------------------------------------------------------
# Load tissue-specific labels (for differential analysis only)
# ---------------------------------------------------------------------------

def load_tissue_labels(
    tissue_labels_path: str,
    windows_per_gene: np.ndarray,
    annot_binary_window: np.ndarray | None = None,
) -> dict[str, list[np.ndarray]]:
    """Load tissue-specific binary labels and stitch to gene level.

    If annot_binary_window is provided, tissue labels are intersected with
    annotation labels so that only annotation-confirmed splice sites that
    are also active in a tissue count as positive.

    Returns dict: tissue -> list of (gene_len,) int32 arrays.
    """
    tissue_gene_labels = {}
    with h5py.File(tissue_labels_path, "r") as f:
        for key in f.keys():
            if not key.endswith("_labels"):
                continue
            tissue = key.replace("_labels", "")
            labels = f[key][:]  # (total_windows, 5000)

            if annot_binary_window is not None:
                # Only keep positions present in BOTH tissue AND annotations
                labels = labels & annot_binary_window

            gene_labels = stitch_gene_labels(labels, windows_per_gene)
            tissue_gene_labels[tissue] = gene_labels
            n_splice = sum(l.sum() for l in gene_labels)
            print(f"  {tissue}: {n_splice} splice positions across "
                  f"{len(gene_labels)} genes")

    return tissue_gene_labels


# ---------------------------------------------------------------------------
# Load model predictions
# ---------------------------------------------------------------------------

def load_splicemamba_preds(
    checkpoint_path: str,
    cfg: dict,
    device: torch.device,
) -> list[np.ndarray]:
    """Run SpliceMamba inference and return gene-level binary P(splice)."""
    from evaluate import load_model, predict_windows

    model = load_model(checkpoint_path, cfg, device)
    all_probs = predict_windows(model, cfg["test_dataset_path"], cfg, device)
    del model
    torch.cuda.empty_cache()

    windows_per_gene = compute_gene_window_counts(cfg["test_datafile_path"])
    gene_probs_3class = stitch_gene_predictions(all_probs, windows_per_gene)
    return adapt_to_binary_splice(gene_probs_3class)


def load_pangolin_preds(
    preds_path: str,
    windows_per_gene: np.ndarray,
) -> dict[str, list[np.ndarray]]:
    """Load pre-computed Pangolin predictions from .npz file.

    Returns dict: tissue -> list of (gene_len,) float32 arrays.
    """
    data = np.load(preds_path)
    preds = {}
    for tissue in data.files:
        arr = data[tissue]
        if arr.ndim == 2:
            # Window-level format (total_windows, 5000) — stitch to gene-level
            preds[tissue] = stitch_gene_predictions(arr, windows_per_gene)
        else:
            # Legacy flat format (total_positions,) — split manually
            gene_probs = []
            offset = 0
            for n_win in windows_per_gene:
                n_win = int(n_win)
                gene_len = n_win * 5000
                gene_probs.append(arr[offset:offset + gene_len])
                offset += gene_len
            preds[tissue] = gene_probs
    return preds


def load_spliceai_preds(
    preds_path: str,
    windows_per_gene: np.ndarray,
) -> list[np.ndarray]:
    """Load pre-computed SpliceAI predictions from .npz file.

    Returns gene-level binary P(splice) = P(acceptor) + P(donor).
    """
    data = np.load(preds_path)
    all_probs = data["probs"]  # (total_windows, 5000, 3)
    gene_probs_3class = stitch_gene_predictions(all_probs, windows_per_gene)
    return adapt_to_binary_splice(gene_probs_3class)


# ---------------------------------------------------------------------------
# Evaluate: Annotation-based (Pangolin paper methodology)
# ---------------------------------------------------------------------------

def evaluate_annotation_based(
    model_name: str,
    gene_probs: list[np.ndarray] | dict[str, list[np.ndarray]],
    gene_labels_binary: list[np.ndarray],
    cfg: dict,
) -> dict:
    """Evaluate model against annotation-based labels.

    For tissue-specific models (Pangolin), each tissue's predictions are
    scored against the SAME annotation labels. For tissue-agnostic models,
    same predictions are used (one evaluation, not per-tissue).

    This matches the Pangolin paper methodology.
    """
    results = {}

    if isinstance(gene_probs, dict):
        # Tissue-specific model: evaluate each tissue's predictions
        for tissue, probs in gene_probs.items():
            auprc = compute_binary_auprc(probs, gene_labels_binary)
            topk = compute_binary_topk(probs, gene_labels_binary)
            f1 = compute_binary_f1(probs, gene_labels_binary)
            pos = compute_binary_positional(
                probs, gene_labels_binary,
                peak_height=cfg["peak_height"],
                peak_distance=cfg["peak_distance"],
            )
            results[tissue] = {
                "auprc": auprc,
                "topk": topk,
                "f1": f1,
                "positional": pos,
            }
            print(f"  {tissue:>8}: AUPRC={auprc['auprc_splice']:.4f}  "
                  f"Top-k={topk['topk_global_splice']:.4f}  "
                  f"F1={f1['f1_splice_best']:.4f}")

        # Also compute averaged across tissues
        if len(gene_probs) > 1:
            avg_probs = []
            tissues = list(gene_probs.keys())
            for g_idx in range(len(gene_labels_binary)):
                avg = np.mean(
                    [gene_probs[t][g_idx] for t in tissues], axis=0
                )
                avg_probs.append(avg)
            auprc = compute_binary_auprc(avg_probs, gene_labels_binary)
            topk = compute_binary_topk(avg_probs, gene_labels_binary)
            f1 = compute_binary_f1(avg_probs, gene_labels_binary)
            pos = compute_binary_positional(
                avg_probs, gene_labels_binary,
                peak_height=cfg["peak_height"],
                peak_distance=cfg["peak_distance"],
            )
            results["averaged"] = {
                "auprc": auprc,
                "topk": topk,
                "f1": f1,
                "positional": pos,
            }
            print(f"  {'avg':>8}: AUPRC={auprc['auprc_splice']:.4f}  "
                  f"Top-k={topk['topk_global_splice']:.4f}  "
                  f"F1={f1['f1_splice_best']:.4f}")
    else:
        # Tissue-agnostic model: single evaluation
        auprc = compute_binary_auprc(gene_probs, gene_labels_binary)
        topk = compute_binary_topk(gene_probs, gene_labels_binary)
        f1 = compute_binary_f1(gene_probs, gene_labels_binary)
        pos = compute_binary_positional(
            gene_probs, gene_labels_binary,
            peak_height=cfg["peak_height"],
            peak_distance=cfg["peak_distance"],
        )
        results["all"] = {
            "auprc": auprc,
            "topk": topk,
            "f1": f1,
            "positional": pos,
        }
        print(f"  {'all':>8}: AUPRC={auprc['auprc_splice']:.4f}  "
              f"Top-k={topk['topk_global_splice']:.4f}  "
              f"F1={f1['f1_splice_best']:.4f}")

    return results


# ---------------------------------------------------------------------------
# Tissue-differential analysis
# ---------------------------------------------------------------------------

def compute_tissue_differential(
    gene_probs: list[np.ndarray] | dict[str, list[np.ndarray]],
    tissue_gene_labels: dict[str, list[np.ndarray]],
    tissues: list[str],
) -> dict:
    """Analyze model performance on tissue-differential splice sites.

    For each pair of tissues (A, B), find sites that are active in A but
    not in B. Measure: does the model assign high scores to these sites?
    """
    results = {}
    tissue_list = [t for t in tissues if t in tissue_gene_labels]

    for i, tissue_a in enumerate(tissue_list):
        for j, tissue_b in enumerate(tissue_list):
            if i == j:
                continue

            # Find sites active in A but not B
            diff_scores = []
            for g_idx in range(len(tissue_gene_labels[tissue_a])):
                lab_a = tissue_gene_labels[tissue_a][g_idx]
                lab_b = tissue_gene_labels[tissue_b][g_idx]

                # Sites in A but not B
                diff_mask = (lab_a > 0) & (lab_b == 0)
                if diff_mask.sum() == 0:
                    continue

                if isinstance(gene_probs, dict):
                    probs = gene_probs.get(tissue_a, gene_probs.get("averaged"))
                else:
                    probs = gene_probs

                diff_scores.extend(probs[g_idx][diff_mask].tolist())

            if diff_scores:
                arr = np.array(diff_scores)
                key = f"{tissue_a}_not_{tissue_b}"
                results[key] = {
                    "n_sites": len(arr),
                    "mean_score": float(arr.mean()),
                    "recall_at_0.3": float((arr >= 0.3).mean()),
                    "recall_at_0.5": float((arr >= 0.5).mean()),
                }

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    cfg = CONFIG
    if args.tissue_labels:
        cfg["tissue_labels_path"] = args.tissue_labels
    start_time = time.time()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    windows_per_gene = compute_gene_window_counts(cfg["test_datafile_path"])

    # Read annotation labels (ground truth for all evaluations)
    print("Reading annotation labels...")
    annot_window_labels = read_window_labels(cfg["test_dataset_path"])
    annot_binary_window = (annot_window_labels > 0).astype(np.int8)
    gene_labels_3class = stitch_gene_labels(annot_window_labels, windows_per_gene)
    gene_labels_binary = labels_to_binary(gene_labels_3class)
    n_splice = sum(l.sum() for l in gene_labels_binary)
    print(f"  {n_splice} annotation splice sites across {len(gene_labels_binary)} genes")

    # Load tissue-specific GTEx labels (for differential analysis)
    tissue_gene_labels = None
    if Path(cfg["tissue_labels_path"]).exists():
        print("Loading GTEx tissue labels (for differential analysis)...")
        tissue_gene_labels = load_tissue_labels(
            cfg["tissue_labels_path"], windows_per_gene,
            annot_binary_window=annot_binary_window,
        )
        tissues_available = list(tissue_gene_labels.keys())
    else:
        tissues_available = []
        print("No GTEx tissue labels found — skipping differential analysis")

    all_results = {
        "timestamp": datetime.now().isoformat(),
        "evaluation_method": "annotation-based (Pangolin paper methodology)",
        "n_annotation_splice_sites": int(n_splice),
        "tissues_available": tissues_available,
        "models": {},
    }

    # --- SpliceMamba ---
    if args.splicemamba_ckpt:
        print(f"\n{'='*60}")
        print("SPLICEMAMBA — Annotation-Based Evaluation")
        print(f"{'='*60}")
        sm_probs = load_splicemamba_preds(args.splicemamba_ckpt, cfg, device)

        sm_annot = evaluate_annotation_based(
            "splicemamba", sm_probs, gene_labels_binary, cfg
        )
        sm_diff = {}
        if tissue_gene_labels:
            sm_diff = compute_tissue_differential(
                sm_probs, tissue_gene_labels, tissues_available
            )

        all_results["models"]["splicemamba"] = {
            "annotation_based": sm_annot,
            "differential": sm_diff,
        }

    # --- Pangolin ---
    if args.pangolin_preds:
        print(f"\n{'='*60}")
        print("PANGOLIN — Annotation-Based Evaluation (per-tissue models)")
        print(f"{'='*60}")
        pg_probs = load_pangolin_preds(args.pangolin_preds, windows_per_gene)

        pg_annot = evaluate_annotation_based(
            "pangolin", pg_probs, gene_labels_binary, cfg
        )
        pg_diff = {}
        if tissue_gene_labels:
            pg_diff = compute_tissue_differential(
                pg_probs, tissue_gene_labels, tissues_available
            )

        all_results["models"]["pangolin"] = {
            "annotation_based": pg_annot,
            "differential": pg_diff,
        }

    # --- SpliceAI ---
    if args.spliceai_preds:
        print(f"\n{'='*60}")
        print("SPLICEAI — Annotation-Based Evaluation")
        print(f"{'='*60}")
        sa_probs = load_spliceai_preds(args.spliceai_preds, windows_per_gene)

        sa_annot = evaluate_annotation_based(
            "spliceai", sa_probs, gene_labels_binary, cfg
        )
        sa_diff = {}
        if tissue_gene_labels:
            sa_diff = compute_tissue_differential(
                sa_probs, tissue_gene_labels, tissues_available
            )

        all_results["models"]["spliceai"] = {
            "annotation_based": sa_annot,
            "differential": sa_diff,
        }

    elapsed = time.time() - start_time

    # --- Summary table ---
    print(f"\n{'='*60}")
    print("SUMMARY: Annotation-Based Binary Splice Metrics")
    print(f"{'='*60}")
    print(f"{'Model':<15} {'Tissue/Mode':<12} {'AUPRC':>8} {'Top-k':>8} {'F1':>8}")
    print("-" * 55)

    for model_name, model_data in all_results["models"].items():
        annot = model_data["annotation_based"]
        for key, metrics in annot.items():
            auprc = metrics["auprc"]["auprc_splice"]
            topk = metrics["topk"]["topk_global_splice"]
            f1 = metrics["f1"]["f1_splice_best"]
            print(f"{model_name:<15} {key:<12} {auprc:>8.4f} {topk:>8.4f} {f1:>8.4f}")

    print(f"\nTotal time: {elapsed:.1f}s")

    # Save results
    out_path = Path(args.output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    json_path = out_path / "tissue_specific_results.json"
    with open(json_path, "w") as fp:
        json.dump(make_serializable(all_results), fp, indent=2)
    print(f"\nResults saved to {json_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Tissue-specific splice site evaluation"
    )
    parser.add_argument(
        "--splicemamba-ckpt", type=str, default=None,
        help="SpliceMamba checkpoint path",
    )
    parser.add_argument(
        "--pangolin-preds", type=str, default=None,
        help="Path to Pangolin predictions .npz (from evaluate_pangolin.py)",
    )
    parser.add_argument(
        "--spliceai-preds", type=str, default=None,
        help="Path to SpliceAI predictions .npz",
    )
    parser.add_argument(
        "--tissue-labels", type=str, default=None,
        help="Path to tissue labels HDF5 (default: gtex_tissue_labels_chr1.h5)",
    )
    parser.add_argument(
        "--output-dir", type=str,
        default=str(Path(__file__).resolve().parent / "results"),
        help="Output directory (default: results/)",
    )
    args = parser.parse_args()
    main(args)
