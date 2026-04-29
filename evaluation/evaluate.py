"""
SpliceMamba evaluation script.

Runs inference on the test set (chr1, 1,652 genes), performs gene-level
stitching, and computes all metrics from SPEC.md Section 7.

Usage:
    python evaluate.py --checkpoint checkpoints/best.pt
    python evaluate.py --checkpoint checkpoints/last.pt
    python evaluate.py --checkpoint dummy --checkpoints ckpt1.pt ckpt2.pt --save-preds
    python evaluate.py --checkpoint dummy --load-preds results/splicemamba_ensemble_preds.npz
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

# Allow imports from repo root (model.py lives there)
_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import h5py
import numpy as np
import torch
from scipy.optimize import minimize_scalar

from model import SpliceMamba
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score

from eval_utils import (
    compute_gene_window_counts,
    stitch_gene_predictions,
    read_window_labels,
    stitch_gene_labels,
    adapt_to_binary_splice,
    labels_to_binary,
    compute_auprc,
    compute_roc,
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
# Configuration (must match training)
# ---------------------------------------------------------------------------

EVAL_CONFIG = dict(
    test_dataset_path=str(Path(_REPO_ROOT) / "dataset_test_0.h5"),
    test_datafile_path=str(Path(_REPO_ROOT) / "datafile_test_0.h5"),
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
    drop_path_rate=0.0,  # disabled at inference
    n_classes=3,
    max_len=15000,
    label_start=5000,
    label_end=10000,
    batch_size=4,
    peak_height=0.5,
    peak_distance=20,
)


# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------

def load_model(checkpoint_path: str, cfg: dict, device: torch.device) -> torch.nn.Module:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Use saved config for model architecture params if available
    saved_cfg = ckpt.get("config", {})
    model_keys = ["d_model", "n_mamba_layers", "d_state", "expand", "d_conv",
                  "headdim", "n_attn_layers", "n_heads", "window_radius",
                  "n_classes", "max_len"]
    model_cfg = {}
    for key in model_keys:
        model_cfg[key] = saved_cfg.get(key, cfg[key])

    model = SpliceMamba(
        d_model=model_cfg["d_model"],
        n_mamba_layers=model_cfg["n_mamba_layers"],
        d_state=model_cfg["d_state"],
        expand=model_cfg["expand"],
        d_conv=model_cfg["d_conv"],
        headdim=model_cfg["headdim"],
        n_attn_layers=model_cfg["n_attn_layers"],
        n_heads=model_cfg["n_heads"],
        window_radius=model_cfg["window_radius"],
        dropout=cfg.get("dropout", 0.15),
        drop_path_rate=0.0,
        n_classes=model_cfg["n_classes"],
        max_len=model_cfg["max_len"],
    ).to(device)

    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Loaded checkpoint from epoch {ckpt.get('epoch', '?')}, "
          f"best AUPRC: {ckpt.get('best_auprc', '?')}")
    if saved_cfg:
        print(f"  Config from checkpoint: d_model={model_cfg['d_model']}, "
              f"n_mamba={model_cfg['n_mamba_layers']}, "
              f"n_attn={model_cfg['n_attn_layers']}, "
              f"window={model_cfg['window_radius']}")
    return model


# ---------------------------------------------------------------------------
# Per-window inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def predict_windows(
    model: torch.nn.Module,
    dataset_path: str,
    cfg: dict,
    device: torch.device,
    temperature: float = 1.0,
) -> np.ndarray:
    """Run inference on all windows, return softmax probs for label regions.

    Returns array of shape (total_windows, 5000, 3) as float32.
    """
    with h5py.File(dataset_path, "r") as f:
        x_keys = sorted(
            [k for k in f.keys() if k.startswith("X")],
            key=lambda k: int(k[1:]),
        )
        all_probs = []

        for x_key in x_keys:
            shard_num = int(x_key[1:])
            x_data = f[x_key][:]  # (N, 15000, 4) int8
            n_windows = x_data.shape[0]

            shard_probs = []
            for start in range(0, n_windows, cfg["batch_size"]):
                end = min(start + cfg["batch_size"], n_windows)
                batch = torch.from_numpy(
                    x_data[start:end].astype(np.float32)
                ).permute(0, 2, 1).to(device)  # (B, 4, 15000)

                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    _, refined_logits, _ = model(batch)

                # Softmax on label region (with optional temperature scaling)
                label_logits = refined_logits[
                    :, cfg["label_start"]:cfg["label_end"], :
                ]
                probs = torch.softmax(label_logits.float() / temperature, dim=-1)
                shard_probs.append(probs.cpu().numpy())

            all_probs.append(np.concatenate(shard_probs, axis=0))

    return np.concatenate(all_probs, axis=0)  # (total_windows, 5000, 3)


@torch.no_grad()
def predict_windows_ensemble(
    checkpoints: list[str],
    dataset_path: str,
    cfg: dict,
    device: torch.device,
    temperature: float = 1.0,
) -> np.ndarray:
    """Run ensemble inference: average softmax probs across multiple models.

    Returns array of shape (total_windows, 5000, 3) as float32.
    """
    all_probs = None
    n_models = len(checkpoints)

    for i, ckpt_path in enumerate(checkpoints):
        print(f"  Ensemble model {i+1}/{n_models}: {ckpt_path}")
        model = load_model(ckpt_path, cfg, device)
        probs = predict_windows(model, dataset_path, cfg, device, temperature)
        if all_probs is None:
            all_probs = probs
        else:
            all_probs += probs
        del model
        torch.cuda.empty_cache()

    return all_probs / n_models


def calibrate_temperature(
    gene_probs_logits: list[np.ndarray],
    gene_labels: list[np.ndarray],
) -> float:
    """Learn optimal temperature T by minimizing NLL on gene-level data.

    gene_probs_logits: list of (N, 3) arrays of RAW LOGITS (not softmaxed).
    Returns optimal temperature T.
    """
    all_logits = np.concatenate(gene_probs_logits, axis=0)
    all_labels = np.concatenate(gene_labels, axis=0)

    def nll_at_temp(T):
        scaled = all_logits / T
        log_sum_exp = np.log(np.sum(np.exp(scaled - scaled.max(axis=-1, keepdims=True)), axis=-1) + 1e-12) + scaled.max(axis=-1)
        log_probs = scaled - log_sum_exp[:, None]
        nll = -log_probs[np.arange(len(all_labels)), all_labels].mean()
        return nll

    result = minimize_scalar(nll_at_temp, bounds=(0.1, 5.0), method="bounded")
    print(f"  Optimal temperature: {result.x:.3f} (NLL: {result.fun:.4f})")
    return float(result.x)


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def _plot_curves(
    gene_probs: list[np.ndarray],
    gene_labels: list[np.ndarray],
    sweep: dict,
    roc_data: dict,
    out_path: Path,
):
    """Generate PR curve, ROC curve, and threshold sweep plots."""
    all_probs = np.concatenate(gene_probs, axis=0)
    all_labels = np.concatenate(gene_labels, axis=0)

    colors = {"donor": "#1f77b4", "acceptor": "#ff7f0e"}

    # --- Precision-Recall Curve ---
    fig, ax = plt.subplots(figsize=(7, 6))
    for cls_name, cls_idx in [("donor", 2), ("acceptor", 1)]:
        true_binary = (all_labels == cls_idx).astype(np.int32)
        scores = all_probs[:, cls_idx]
        prec, rec, _ = precision_recall_curve(true_binary, scores)
        ap = average_precision_score(true_binary, scores)
        ax.plot(rec, prec, color=colors[cls_name],
                label=f"{cls_name.capitalize()} (AP={ap:.4f})")
        # Mark threshold sweep points
        for row in sweep[cls_name]:
            ax.plot(row["recall"], row["precision"], "o",
                    color=colors[cls_name], markersize=3, alpha=0.6)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend()
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path / "pr_curve.png", dpi=150)
    plt.close(fig)
    print(f"  Saved {out_path / 'pr_curve.png'}")

    # --- ROC Curve ---
    fig, ax = plt.subplots(figsize=(7, 6))
    for cls_name in ["donor", "acceptor"]:
        fpr = roc_data[f"roc_{cls_name}"]["fpr"]
        tpr = roc_data[f"roc_{cls_name}"]["tpr"]
        auroc = roc_data[f"auroc_{cls_name}"]
        ax.plot(fpr, tpr, color=colors[cls_name],
                label=f"{cls_name.capitalize()} (AUROC={auroc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend()
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path / "roc_curve.png", dpi=150)
    plt.close(fig)
    print(f"  Saved {out_path / 'roc_curve.png'}")

    # --- Threshold Sweep (F1, Precision, Recall vs Threshold) ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for ax, cls_name in zip(axes, ["donor", "acceptor"]):
        rows = sweep[cls_name]
        threshs = [r["threshold"] for r in rows]
        ax.plot(threshs, [r["precision"] for r in rows], "s-",
                label="Precision", markersize=3)
        ax.plot(threshs, [r["recall"] for r in rows], "^-",
                label="Recall", markersize=3)
        ax.plot(threshs, [r["f1"] for r in rows], "o-",
                label="F1", markersize=3)
        ax.set_xlabel("Threshold")
        ax.set_ylabel("Score")
        ax.set_title(f"{cls_name.capitalize()} — Threshold Sweep")
        ax.legend()
        ax.set_xlim([0, 1.02])
        ax.set_ylim([0, 1.05])
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path / "threshold_sweep.png", dpi=150)
    plt.close(fig)
    print(f"  Saved {out_path / 'threshold_sweep.png'}")


def evaluate(
    checkpoint_path: str | list[str],
    cfg: dict,
    output_dir: str = str(Path(__file__).resolve().parent / "results"),
    temperature: float = 1.0,
    save_preds: bool = False,
    load_preds: str | None = None,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    is_ensemble = isinstance(checkpoint_path, list) and len(checkpoint_path) > 1

    # Run inference on all test windows (or load cached predictions)
    if load_preds:
        print(f"Loading cached predictions from {load_preds}...")
        all_probs = np.load(load_preds)["probs"]
        print(f"  Loaded {all_probs.shape[0]} windows from cache")
    elif is_ensemble:
        print(f"Running ensemble inference with {len(checkpoint_path)} models...")
        all_probs = predict_windows_ensemble(
            checkpoint_path, cfg["test_dataset_path"], cfg, device, temperature
        )
    else:
        # Single model
        ckpt = checkpoint_path[0] if isinstance(checkpoint_path, list) else checkpoint_path
        model = load_model(ckpt, cfg, device)
        print("Running inference on test set...")
        all_probs = predict_windows(model, cfg["test_dataset_path"], cfg, device, temperature)
        del model

    # Save predictions if requested
    if save_preds and not load_preds:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        tag = "ensemble" if is_ensemble else "single"
        preds_path = out_path / f"splicemamba_{tag}_preds.npz"
        np.savez_compressed(preds_path, probs=all_probs)
        print(f"  Predictions saved to {preds_path}")

    print(f"  Predicted {all_probs.shape[0]} windows")

    # Read preprocessed labels
    print("Reading preprocessed labels...")
    all_labels = read_window_labels(cfg["test_dataset_path"])
    print(f"  Read {all_labels.shape[0]} window labels")

    # Gene-level stitching
    print("Stitching gene-level predictions...")
    windows_per_gene = compute_gene_window_counts(cfg["test_datafile_path"])
    gene_probs = stitch_gene_predictions(all_probs, windows_per_gene)
    gene_labels = stitch_gene_labels(all_labels, windows_per_gene)
    print(f"  Stitched {len(gene_probs)} genes")

    # Compute all metrics
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)

    # AUPRC
    auprc = compute_auprc(gene_probs, gene_labels)
    print(f"\nAUPRC:")
    print(f"  Donor:    {auprc['auprc_donor']:.4f}")
    print(f"  Acceptor: {auprc['auprc_acceptor']:.4f}")
    print(f"  Mean:     {auprc['auprc_mean']:.4f}")

    # Top-k accuracy
    topk = compute_topk_accuracy(gene_probs, gene_labels)
    print(f"\nTop-k Accuracy (Global — SpliceAI method):")
    print(f"  Donor:    {topk['topk_global_donor']:.4f}")
    print(f"  Acceptor: {topk['topk_global_acceptor']:.4f}")
    print(f"  Mean:     {topk['topk_global_mean']:.4f}  (SpliceAI reports 0.95)")
    print(f"\nTop-k Accuracy (Per-gene average):")
    for cls_name in ["donor", "acceptor"]:
        print(f"  {cls_name.capitalize()}:")
        for k in [0.5, 1, 2, 4]:
            val = topk[f"topk_{cls_name}_k{k}"]
            print(f"    k={k}: {val:.4f}")

    # Positional accuracy
    pos = compute_positional_accuracy(
        gene_probs, gene_labels,
        peak_height=cfg["peak_height"],
        peak_distance=cfg["peak_distance"],
    )
    print(f"\nPositional Accuracy:")
    for cls_name in ["donor", "acceptor"]:
        print(f"  {cls_name.capitalize()}:")
        print(f"    Mean offset:  {pos[f'positional_{cls_name}_mean_offset']:.2f} bp")
        print(f"    Median offset: {pos[f'positional_{cls_name}_median_offset']:.1f} bp")
        print(f"    Within ±1 bp: {pos[f'positional_{cls_name}_within_1bp']:.1%}")
        print(f"    Within ±5 bp: {pos[f'positional_{cls_name}_within_5bp']:.1%}")

    # F1 at optimal threshold
    f1 = compute_f1_at_optimal_threshold(gene_probs, gene_labels)
    print(f"\nF1 at Optimal Threshold:")
    for cls_name in ["donor", "acceptor"]:
        print(f"  {cls_name.capitalize()}: F1={f1[f'f1_{cls_name}_best']:.4f} "
              f"@ threshold={f1[f'f1_{cls_name}_threshold']:.2f}")

    # Threshold sweep
    print(f"\nThreshold Sweep:")
    sweep = compute_threshold_sweep(gene_probs, gene_labels)
    for cls_name in ["donor", "acceptor"]:
        print(f"  {cls_name.capitalize()}:")
        print(f"    {'Thresh':>6}  {'Prec':>6}  {'Recall':>6}  {'F1':>6}")
        for row in sweep[cls_name]:
            print(f"    {row['threshold']:>6.2f}  {row['precision']:>6.4f}  "
                  f"{row['recall']:>6.4f}  {row['f1']:>6.4f}")

    # ROC
    roc_data = compute_roc(gene_probs, gene_labels)
    print(f"\nAUROC:")
    print(f"  Donor:    {roc_data['auroc_donor']:.4f}")
    print(f"  Acceptor: {roc_data['auroc_acceptor']:.4f}")
    print(f"  Mean:     {roc_data['auroc_mean']:.4f}")

    # Stratified metrics by intron/exon length
    print("\nParsing gene junctions for stratified analysis...")
    genes = parse_gene_junctions(cfg["test_datafile_path"])
    stratified = compute_stratified_metrics(gene_probs, gene_labels, genes)

    for strat_key, strat_label in [
        ("by_intron_length", "Intron Length"),
        ("by_exon_length", "Exon Length"),
    ]:
        print(f"\nStratified by {strat_label}:")
        print(f"  {'Bucket':<15} {'Class':<10} {'N':>6} {'Recall@0.3':>10} "
              f"{'Recall@0.5':>10} {'Recall@0.7':>10} {'Mean Score':>10}")
        print("  " + "-" * 75)
        for bname, bdata in stratified[strat_key].items():
            for cls_name in ["donor", "acceptor"]:
                n_key = f"{cls_name}_n_sites"
                if n_key not in bdata:
                    continue
                n = bdata[n_key]
                r3 = bdata[f"{cls_name}_recall_at_0.3"]
                r5 = bdata[f"{cls_name}_recall_at_0.5"]
                r7 = bdata[f"{cls_name}_recall_at_0.7"]
                ms = bdata[f"{cls_name}_mean_score"]
                print(f"  {bname:<15} {cls_name:<10} {n:>6} {r3:>10.4f} "
                      f"{r5:>10.4f} {r7:>10.4f} {ms:>10.4f}")

    # Binary splice metrics (for Pangolin comparison)
    gene_probs_binary = adapt_to_binary_splice(gene_probs)
    gene_labels_bin = labels_to_binary(gene_labels)
    binary_auprc = compute_binary_auprc(gene_probs_binary, gene_labels_bin)
    binary_topk = compute_binary_topk(gene_probs_binary, gene_labels_bin)
    binary_f1 = compute_binary_f1(gene_probs_binary, gene_labels_bin)
    binary_pos = compute_binary_positional(
        gene_probs_binary, gene_labels_bin,
        peak_height=cfg["peak_height"],
        peak_distance=cfg["peak_distance"],
    )
    print(f"\nBinary Splice Metrics (for Pangolin comparison):")
    print(f"  AUPRC: {binary_auprc['auprc_splice']:.4f}")
    print(f"  Top-k: {binary_topk['topk_global_splice']:.4f}")
    print(f"  F1:    {binary_f1['f1_splice_best']:.4f} "
          f"@ threshold={binary_f1['f1_splice_threshold']:.2f}")
    print(f"  Within ±1bp: {binary_pos['positional_splice_within_1bp']:.1%}")

    print("\n" + "=" * 60)

    # Generate plots
    print("\nGenerating plots...")
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    _plot_curves(gene_probs, gene_labels, sweep, roc_data, out_path)

    # Save results to JSON
    all_results = {
        "model": "splicemamba" + ("_ensemble" if is_ensemble else ""),
        "checkpoint": [str(c) for c in checkpoint_path] if isinstance(checkpoint_path, list) else str(checkpoint_path),
        "temperature": temperature,
        "timestamp": datetime.now().isoformat(),
        "n_genes": len(gene_probs),
        "metrics": {
            "auprc": auprc,
            "auroc": {
                "auroc_donor": roc_data["auroc_donor"],
                "auroc_acceptor": roc_data["auroc_acceptor"],
                "auroc_mean": roc_data["auroc_mean"],
            },
            "topk": topk,
            "positional": pos,
            "f1_optimal": f1,
            "threshold_sweep": sweep,
            "stratified_by_intron_length": stratified["by_intron_length"],
            "stratified_by_exon_length": stratified["by_exon_length"],
            "binary": {
                "auprc": binary_auprc,
                "topk": binary_topk,
                "f1": binary_f1,
                "positional": binary_pos,
            },
        },
    }

    json_path = out_path / "splicemamba_results.json"
    with open(json_path, "w") as fp:
        json.dump(make_serializable(all_results), fp, indent=2)
    print(f"\nResults saved to {json_path}")

    return all_results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate SpliceMamba")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--checkpoints", type=str, nargs="+", default=None,
                        help="Multiple checkpoint paths for ensemble evaluation")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Temperature scaling for softmax (default: 1.0)")
    parser.add_argument("--output-dir", type=str,
                        default=str(Path(__file__).resolve().parent / "results"),
                        help="Directory to save results JSON")
    parser.add_argument("--save-preds", action="store_true",
                        help="Save raw predictions as .npz for reuse")
    parser.add_argument("--load-preds", type=str, default=None,
                        help="Load predictions from .npz instead of running inference")
    args = parser.parse_args()

    ckpts = args.checkpoints if args.checkpoints else args.checkpoint
    evaluate(ckpts, EVAL_CONFIG, output_dir=args.output_dir,
             temperature=args.temperature,
             save_preds=args.save_preds,
             load_preds=args.load_preds)
