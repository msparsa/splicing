"""
Evaluate pre-trained SpliceAI (10k context, 5-model ensemble) on the test set.

Metrics are computed identically to evaluate.py so results are directly
comparable with SpliceMamba.

Usage:
    /mnt/lareaulab/mparsa/miniconda3/envs/spliceai_env/bin/python evaluate_spliceai.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['PYTHONUNBUFFERED'] = '1'

import time
from datetime import datetime
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(f"Using GPU: {[g.name for g in gpus]}")
else:
    print("WARNING: No GPU found, running on CPU")
from math import ceil

import h5py
import numpy as np
from pkg_resources import resource_filename

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
# Configuration — matches evaluate.py
# ---------------------------------------------------------------------------

CONFIG = dict(
    test_dataset_path=str(Path(_REPO_ROOT) / "dataset_test_0.h5"),
    test_datafile_path=str(Path(_REPO_ROOT) / "datafile_test_0.h5"),
    CL=10000,
    SL=5000,
    batch_size=6,
    peak_height=0.5,
    peak_distance=20,
)


# ---------------------------------------------------------------------------
# Load SpliceAI models
# ---------------------------------------------------------------------------

def load_spliceai_models(n_models=5):
    """Load pre-trained SpliceAI 10k models from the pip package."""
    from keras.models import load_model

    models = []
    for i in range(1, n_models + 1):
        path = resource_filename("spliceai", f"models/spliceai{i}.h5")
        print(f"  Loading {path} ...")
        models.append(load_model(path, compile=False))
    return models


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def predict_windows(models, dataset_path, cfg):
    """Run ensemble inference on all test windows.

    Averages softmax outputs across all models.
    Returns array of shape (total_windows, 5000, 3) as float32.
    """
    CL = cfg["CL"]
    CL_max = 10000
    clip = (CL_max - CL) // 2  # 0 for 10k model

    with h5py.File(dataset_path, "r") as f:
        x_keys = sorted(
            [k for k in f.keys() if k.startswith("X")],
            key=lambda k: int(k[1:]),
        )
        all_probs = []

        for x_key in x_keys:
            print(f"  Processing shard {x_key} ...")
            x_data = f[x_key][:]  # (N, 15000, 4) int8

            # Clip context if needed (no-op for CL=10000)
            if clip > 0:
                x_data = x_data[:, clip:-clip, :]

            x_data = x_data.astype(np.float32)

            # Average predictions across ensemble members
            shard_preds = np.zeros((x_data.shape[0], 5000, 3), dtype=np.float32)
            for model in models:
                preds = model.predict(x_data, batch_size=cfg["batch_size"],
                                      verbose=0)
                shard_preds += preds
            shard_preds /= len(models)

            all_probs.append(shard_preds)

    return np.concatenate(all_probs, axis=0)  # (total_windows, 5000, 3)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(output_dir="results", save_preds=False):
    cfg = CONFIG
    start_time = time.time()

    # Load models
    print("Loading SpliceAI 10k models (5-model ensemble)...")
    models = load_spliceai_models(n_models=5)
    print(f"  Loaded {len(models)} models")

    # Run inference
    print("\nRunning ensemble inference on test set...")
    all_probs = predict_windows(models, cfg["test_dataset_path"], cfg)
    print(f"  Predicted {all_probs.shape[0]} windows, shape: {all_probs.shape}")

    # Read labels
    print("\nReading preprocessed labels...")
    all_labels = read_window_labels(cfg["test_dataset_path"])
    print(f"  Read {all_labels.shape[0]} window labels")

    # Gene-level stitching
    print("\nStitching gene-level predictions...")
    windows_per_gene = compute_gene_window_counts(cfg["test_datafile_path"])
    gene_probs = stitch_gene_predictions(all_probs, windows_per_gene)
    gene_labels = stitch_gene_labels(all_labels, windows_per_gene)
    print(f"  Stitched {len(gene_probs)} genes")

    # --- Compute all metrics (same as evaluate.py) ---
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS — SpliceAI 10k (5-model ensemble)")
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

    elapsed = time.time() - start_time
    print(f"\n{'=' * 60}")
    print(f"Total time: {elapsed:.1f}s")

    # Save results to JSON
    all_results = {
        "model": "spliceai",
        "checkpoint": "ensemble-5x10k",
        "timestamp": datetime.now().isoformat(),
        "n_genes": len(gene_probs),
        "metrics": {
            "auprc": auprc,
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

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    json_path = out_path / "spliceai_results.json"
    with open(json_path, "w") as fp:
        json.dump(make_serializable(all_results), fp, indent=2)
    print(f"\nResults saved to {json_path}")

    # Optionally save raw predictions for tissue-specific evaluation
    if save_preds:
        preds_path = out_path / "spliceai_preds.npz"
        np.savez_compressed(preds_path, probs=all_probs)
        print(f"Predictions saved to {preds_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate SpliceAI baseline")
    parser.add_argument("--output-dir", type=str,
                        default=str(Path(__file__).resolve().parent / "results"),
                        help="Directory to save results JSON")
    parser.add_argument("--save-preds", action="store_true",
                        help="Save raw predictions as .npz for tissue evaluation")
    args = parser.parse_args()
    main(output_dir=args.output_dir, save_preds=args.save_preds)
