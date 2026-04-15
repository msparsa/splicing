"""
Evaluate Pangolin on the same chr1 test set used for SpliceMamba/SpliceAI.

Pangolin outputs tissue-specific P(splice) for 4 tissues (Heart, Liver, Brain,
Testis) using a 5-model ensemble per tissue. Its context loss CL=10000, so a
15000bp input produces 5000bp output — perfectly matching the label region.

Usage:
    python evaluate_pangolin.py
    python evaluate_pangolin.py --tissues heart liver
    python evaluate_pangolin.py --output-dir results/pangolin
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
from tqdm import tqdm

from pangolin.model import Pangolin, L, W, AR

from eval_utils import (
    compute_gene_window_counts,
    stitch_gene_predictions,
    read_window_labels,
    stitch_gene_labels,
    adapt_to_binary_splice,
    labels_to_binary,
    compute_auprc,
    compute_topk_accuracy,
    compute_positional_accuracy,
    compute_f1_at_optimal_threshold,
    compute_threshold_sweep,
    parse_gene_junctions,
    compute_stratified_metrics,
    compute_binary_auprc,
    compute_binary_topk,
    compute_binary_f1,
    compute_binary_positional,
    make_serializable,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TISSUE_MODELS = {
    "heart":  {"model_num": 0, "channel_idx": 1,  "label": "Heart P(splice)"},
    "liver":  {"model_num": 2, "channel_idx": 4,  "label": "Liver P(splice)"},
    "brain":  {"model_num": 4, "channel_idx": 7,  "label": "Brain P(splice)"},
    "testis": {"model_num": 6, "channel_idx": 10, "label": "Testis P(splice)"},
}

# Pangolin also has "usage" models (sigmoid output) at model_nums 1,3,5,7
# but we focus on P(splice) models for splice site prediction comparison.

CONFIG = dict(
    test_dataset_path=str(Path(_REPO_ROOT) / "dataset_test_0.h5"),
    test_datafile_path=str(Path(_REPO_ROOT) / "datafile_test_0.h5"),
    batch_size=64,  # Pangolin is 0.7M params, can use large batches
    peak_height=0.3,
    peak_distance=20,
)


# ---------------------------------------------------------------------------
# Load Pangolin models
# ---------------------------------------------------------------------------

def load_pangolin_models(tissue: str, device: torch.device) -> list[torch.nn.Module]:
    """Load 5-model ensemble for a given tissue's P(splice) model.

    Models 1-3 use .v2 weights, models 4-5 use original weights.
    """
    import pangolin
    models_dir = Path(pangolin.__file__).parent / "models"

    model_num = TISSUE_MODELS[tissue]["model_num"]
    models = []
    for j in range(1, 6):
        model = Pangolin(L, W, AR)
        # Models 1-3 have updated v2 weights
        if j <= 3:
            weight_path = models_dir / f"final.{j}.{model_num}.3.v2"
        else:
            weight_path = models_dir / f"final.{j}.{model_num}.3"
        weights = torch.load(str(weight_path), map_location=device, weights_only=False)
        model.load_state_dict(weights)
        model.to(device)
        model.eval()
        models.append(model)
    return models


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def predict_pangolin_all_tissues(
    tissue_models: dict[str, list[torch.nn.Module]],
    dataset_path: str,
    cfg: dict,
    device: torch.device,
) -> dict[str, np.ndarray]:
    """Run Pangolin inference for all tissues in a single data pass.

    Pangolin is only 0.7M params, so we keep all tissue ensembles in memory
    and iterate over the data once, running all models per batch.

    Returns dict: tissue -> (total_windows, 5000) float32 P(splice).
    """
    tissues = list(tissue_models.keys())

    with h5py.File(dataset_path, "r") as f:
        x_keys = sorted(
            [k for k in f.keys() if k.startswith("X")],
            key=lambda k: int(k[1:]),
        )

        # Count total windows for progress bar
        total_windows = sum(f[k].shape[0] for k in x_keys)
        n_tissues = len(tissues)
        n_ensemble = 5

        # Accumulate per-tissue results
        tissue_probs = {t: [] for t in tissues}

        pbar = tqdm(
            total=total_windows,
            desc=f"Pangolin inference ({n_tissues} tissues × {n_ensemble} models)",
            unit="win",
        )

        for x_key in x_keys:
            x_data = f[x_key][:]  # (N, 15000, 4) int8
            n_windows = x_data.shape[0]

            shard_probs = {t: [] for t in tissues}
            for start in range(0, n_windows, cfg["batch_size"]):
                end = min(start + cfg["batch_size"], n_windows)
                batch = torch.from_numpy(
                    x_data[start:end].astype(np.float32)
                ).permute(0, 2, 1).to(device)  # (B, 4, 15000)

                for tissue in tissues:
                    channel_idx = TISSUE_MODELS[tissue]["channel_idx"]
                    models = tissue_models[tissue]

                    # Average across 5 ensemble models
                    scores_sum = None
                    for model in models:
                        out = model(batch)  # (B, 12, 5000)
                        s = out[:, channel_idx, :]  # (B, 5000)
                        if scores_sum is None:
                            scores_sum = s
                        else:
                            scores_sum = scores_sum + s

                    avg = (scores_sum / len(models)).cpu().numpy()
                    shard_probs[tissue].append(avg)

                pbar.update(end - start)

            for tissue in tissues:
                tissue_probs[tissue].append(
                    np.concatenate(shard_probs[tissue], axis=0)
                )

        pbar.close()

    return {t: np.concatenate(v, axis=0) for t, v in tissue_probs.items()}


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def evaluate_pangolin(
    tissues: list[str],
    cfg: dict,
    output_dir: str = "results",
    save_preds: bool = True,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device -> {device} <-")
    start_time = time.time()

    # Read labels (same for all models)
    print("Reading preprocessed labels...")
    all_labels = read_window_labels(cfg["test_dataset_path"])
    windows_per_gene = compute_gene_window_counts(cfg["test_datafile_path"])
    gene_labels_3class = stitch_gene_labels(all_labels, windows_per_gene)
    gene_labels_binary = labels_to_binary(gene_labels_3class)
    print(f"  {len(gene_labels_3class)} genes, "
          f"{sum(l.sum() for l in gene_labels_binary)} splice sites")

    # Parse junctions for stratified metrics
    genes = parse_gene_junctions(cfg["test_datafile_path"])

    # Check for cached predictions
    preds_path = Path(output_dir) / "pangolin_preds.npz"
    if preds_path.exists():
        print(f"\nFound cached predictions at {preds_path}, skipping inference!")
        cached = np.load(preds_path)
        tissue_window_probs = {t: cached[t] for t in tissues if t in cached}
        missing = [t for t in tissues if t not in tissue_window_probs]
        if missing:
            print(f"  Missing tissues in cache: {missing}, running inference for them...")
        else:
            missing = []
    else:
        tissue_window_probs = {}
        missing = tissues

    if missing:
        # Load tissue models and run inference for missing tissues
        print(f"\nLoading tissue ensembles for {missing}...")
        tissue_models = {}
        for tissue in missing:
            print(f"  Loading {tissue} ensemble (5 models)...")
            tissue_models[tissue] = load_pangolin_models(tissue, device)

        print(f"\nRunning inference ({len(missing)} tissues, single data pass)...")
        new_probs = predict_pangolin_all_tissues(
            tissue_models, cfg["test_dataset_path"], cfg, device
        )
        tissue_window_probs.update(new_probs)

        # Free GPU memory
        del tissue_models
        torch.cuda.empty_cache()

    tissue_results = {}
    all_tissue_preds = {}

    for tissue in tissues:
        print(f"\n{'='*60}")
        print(f"PANGOLIN — {TISSUE_MODELS[tissue]['label']}")
        print(f"{'='*60}")

        all_probs = tissue_window_probs[tissue]
        print(f"  {all_probs.shape[0]} windows, shape: {all_probs.shape}")

        # Gene-level stitching (binary: splice vs not-splice)
        gene_probs_splice = stitch_gene_predictions(all_probs, windows_per_gene)
        all_tissue_preds[tissue] = gene_probs_splice

        # --- Binary splice metrics ---
        auprc = compute_binary_auprc(gene_probs_splice, gene_labels_binary)
        topk = compute_binary_topk(gene_probs_splice, gene_labels_binary)
        f1 = compute_binary_f1(gene_probs_splice, gene_labels_binary)
        pos = compute_binary_positional(
            gene_probs_splice, gene_labels_binary,
            peak_height=cfg["peak_height"],
            peak_distance=cfg["peak_distance"],
        )

        print(f"\nBinary Splice AUPRC: {auprc['auprc_splice']:.4f}")
        print(f"Top-k (global): {topk['topk_global_splice']:.4f}")
        print(f"F1 best: {f1['f1_splice_best']:.4f} "
              f"@ threshold={f1['f1_splice_threshold']:.2f}")
        print(f"Positional within ±1bp: "
              f"{pos['positional_splice_within_1bp']:.1%}")

        tissue_results[tissue] = {
            "auprc": auprc,
            "topk": topk,
            "f1": f1,
            "positional": pos,
        }

    # --- Averaged across tissues ---
    if len(tissues) > 1:
        print(f"\n{'='*60}")
        print(f"PANGOLIN — Averaged across {len(tissues)} tissues")
        print(f"{'='*60}")

        # Average P(splice) across tissue models
        avg_gene_probs = []
        for g_idx in range(len(gene_labels_binary)):
            avg = np.mean(
                [all_tissue_preds[t][g_idx] for t in tissues], axis=0
            )
            avg_gene_probs.append(avg)

        auprc_avg = compute_binary_auprc(avg_gene_probs, gene_labels_binary)
        topk_avg = compute_binary_topk(avg_gene_probs, gene_labels_binary)
        f1_avg = compute_binary_f1(avg_gene_probs, gene_labels_binary)
        pos_avg = compute_binary_positional(
            avg_gene_probs, gene_labels_binary,
            peak_height=cfg["peak_height"],
            peak_distance=cfg["peak_distance"],
        )

        print(f"\nBinary Splice AUPRC: {auprc_avg['auprc_splice']:.4f}")
        print(f"Top-k (global): {topk_avg['topk_global_splice']:.4f}")
        print(f"F1 best: {f1_avg['f1_splice_best']:.4f} "
              f"@ threshold={f1_avg['f1_splice_threshold']:.2f}")

        tissue_results["averaged"] = {
            "auprc": auprc_avg,
            "topk": topk_avg,
            "f1": f1_avg,
            "positional": pos_avg,
        }

    elapsed = time.time() - start_time
    print(f"\nTotal time: {elapsed:.1f}s")

    # Save results
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    all_results = {
        "model": "pangolin",
        "tissues_evaluated": tissues,
        "timestamp": datetime.now().isoformat(),
        "n_genes": len(gene_labels_3class),
        "results_per_tissue": tissue_results,
    }

    json_path = out_path / "pangolin_results.json"
    with open(json_path, "w") as fp:
        json.dump(make_serializable(all_results), fp, indent=2)
    print(f"\nResults saved to {json_path}")

    # Save raw predictions for downstream use (tissue-specific eval, etc.)
    # NOTE: Must save window-level arrays (total_windows, 5000) — NOT stitched
    # gene-level flat arrays — so that loading + stitch_gene_predictions works.
    if save_preds:
        preds_path = out_path / "pangolin_preds.npz"
        save_dict = {}
        for tissue in tissues:
            save_dict[tissue] = tissue_window_probs[tissue]  # (total_windows, 5000)
        np.savez_compressed(preds_path, **save_dict)
        print(f"Predictions saved to {preds_path}")

    return all_results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate Pangolin on chr1 test set"
    )
    parser.add_argument(
        "--tissues", nargs="+",
        default=["heart", "liver", "brain", "testis"],
        choices=["heart", "liver", "brain", "testis"],
        help="Tissues to evaluate (default: all 4)",
    )
    parser.add_argument(
        "--output-dir", type=str,
        default=str(Path(__file__).resolve().parent / "results"),
        help="Directory to save results",
    )
    parser.add_argument(
        "--no-save-preds", action="store_true",
        help="Don't save raw predictions as .npz",
    )
    args = parser.parse_args()

    evaluate_pangolin(
        tissues=args.tissues,
        cfg=CONFIG,
        output_dir=args.output_dir,
        save_preds=not args.no_save_preds,
    )
