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
import math
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['PYTHONUNBUFFERED'] = '1'

import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

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
from scipy.signal import find_peaks
from sklearn.metrics import average_precision_score

# ---------------------------------------------------------------------------
# Configuration — matches evaluate.py
# ---------------------------------------------------------------------------

CONFIG = dict(
    test_dataset_path="dataset_test_0.h5",
    test_datafile_path="datafile_test_0.h5",
    CL=10000,
    SL=5000,
    batch_size=6,
    peak_height=0.5,
    peak_distance=20,
    n_bootstrap=1000,
    bootstrap_seed=42,
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
# Gene-level stitching — identical to evaluate.py
# ---------------------------------------------------------------------------

def compute_gene_window_counts(datafile_path):
    """Return per-gene window counts: ceil(gene_len / 5000)."""
    with h5py.File(datafile_path, "r") as f:
        tx_start = f["TX_START"][:]
        tx_end = f["TX_END"][:]
    gene_lens = tx_end - tx_start
    return np.ceil(gene_lens / 5000).astype(np.int64)


def stitch_gene_predictions(all_probs, windows_per_gene):
    """Stitch window predictions into gene-level probability arrays.

    Returns a list of (gene_len_labels, 3) arrays.
    """
    gene_probs = []
    offset = 0
    for n_win in windows_per_gene:
        n_win = int(n_win)
        gene_pred = all_probs[offset : offset + n_win]  # (n_win, 5000, 3)
        gene_pred = gene_pred.reshape(-1, 3)  # (n_win * 5000, 3)
        gene_probs.append(gene_pred)
        offset += n_win
    return gene_probs


def read_window_labels(dataset_path):
    """Read preprocessed Y labels from the dataset HDF5 file.

    Returns array of shape (total_windows, 5000) as int64 class indices.
    """
    with h5py.File(dataset_path, "r") as f:
        y_keys = sorted(
            [k for k in f.keys() if k.startswith("Y")],
            key=lambda k: int(k[1:]),
        )
        all_labels = []
        for y_key in y_keys:
            y_data = f[y_key][0]  # (N, 5000, 3) int8 — squeeze leading dim
            labels = np.argmax(y_data, axis=-1)  # (N, 5000) int64
            all_labels.append(labels)
    return np.concatenate(all_labels, axis=0)  # (total_windows, 5000)


def stitch_gene_labels(all_labels, windows_per_gene):
    """Stitch window-level labels into gene-level label arrays."""
    gene_labels = []
    offset = 0
    for n_win in windows_per_gene:
        n_win = int(n_win)
        gene_lab = all_labels[offset : offset + n_win]  # (n_win, 5000)
        gene_lab = gene_lab.reshape(-1)  # (n_win * 5000,)
        gene_labels.append(gene_lab)
        offset += n_win
    return gene_labels


# ---------------------------------------------------------------------------
# Metrics — copied verbatim from evaluate.py
# ---------------------------------------------------------------------------

def compute_auprc(gene_probs, gene_labels):
    """Compute AUPRC for donor and acceptor classes across all genes."""
    all_probs = np.concatenate(gene_probs, axis=0)  # (N, 3)
    all_labels = np.concatenate(gene_labels, axis=0)  # (N,)

    acc_true = (all_labels == 1).astype(np.int32)
    acc_score = all_probs[:, 1]
    auprc_acceptor = average_precision_score(acc_true, acc_score)

    donor_true = (all_labels == 2).astype(np.int32)
    donor_score = all_probs[:, 2]
    auprc_donor = average_precision_score(donor_true, donor_score)

    auprc_mean = (auprc_donor + auprc_acceptor) / 2.0

    return {
        "auprc_donor": auprc_donor,
        "auprc_acceptor": auprc_acceptor,
        "auprc_mean": auprc_mean,
    }


def compute_topk_accuracy(gene_probs, gene_labels):
    """Compute top-k accuracy both globally (SpliceAI method) and per-gene."""
    all_probs = np.concatenate(gene_probs, axis=0)  # (N, 3)
    all_labels = np.concatenate(gene_labels, axis=0)  # (N,)

    results = {}
    for cls_name, cls_idx in [("acceptor", 1), ("donor", 2)]:
        true_mask = all_labels == cls_idx
        n_true = int(true_mask.sum())
        if n_true == 0:
            results[f"topk_global_{cls_name}"] = 0.0
            continue
        scores = all_probs[:, cls_idx]
        top_idx = np.argpartition(-scores, n_true)[:n_true]
        n_found = true_mask[top_idx].sum()
        results[f"topk_global_{cls_name}"] = float(n_found) / float(n_true)

    results["topk_global_mean"] = (
        results.get("topk_global_donor", 0.0) +
        results.get("topk_global_acceptor", 0.0)
    ) / 2.0

    # --- Per-gene top-k at k = {0.5, 1, 2, 4} ---
    per_gene = {f"topk_{c}_k{k}": [] for c in ["donor", "acceptor"] for k in [0.5, 1, 2, 4]}

    for probs, labels in zip(gene_probs, gene_labels):
        for cls_name, cls_idx in [("acceptor", 1), ("donor", 2)]:
            true_mask = labels == cls_idx
            n_true = true_mask.sum()
            if n_true == 0:
                continue

            scores = probs[:, cls_idx]
            for k in [0.5, 1, 2, 4]:
                n_select = max(1, int(k * n_true))
                top_idx = np.argpartition(-scores, n_select)[:n_select]
                n_found = true_mask[top_idx].sum()
                per_gene[f"topk_{cls_name}_k{k}"].append(float(n_found) / float(n_true))

    for k, v in per_gene.items():
        results[k] = np.mean(v) if v else 0.0

    return results


def compute_positional_accuracy(gene_probs, gene_labels,
                                 peak_height=0.5, peak_distance=20):
    """Compute positional accuracy: offsets between predicted peaks and
    nearest true splice sites."""
    offsets = {"donor": [], "acceptor": []}

    for probs, labels in zip(gene_probs, gene_labels):
        for cls_name, cls_idx in [("acceptor", 1), ("donor", 2)]:
            true_positions = np.where(labels == cls_idx)[0]
            if len(true_positions) == 0:
                continue

            scores = probs[:, cls_idx]
            peaks, _ = find_peaks(scores, height=peak_height, distance=peak_distance)

            if len(peaks) == 0:
                continue

            for tp in true_positions:
                dists = np.abs(peaks.astype(np.int64) - tp)
                offsets[cls_name].append(int(dists.min()))

    results = {}
    for cls_name in ["donor", "acceptor"]:
        if offsets[cls_name]:
            arr = np.array(offsets[cls_name])
            results[f"positional_{cls_name}_mean_offset"] = float(arr.mean())
            results[f"positional_{cls_name}_median_offset"] = float(np.median(arr))
            results[f"positional_{cls_name}_within_1bp"] = float((arr <= 1).mean())
            results[f"positional_{cls_name}_within_5bp"] = float((arr <= 5).mean())
        else:
            results[f"positional_{cls_name}_mean_offset"] = float("inf")
            results[f"positional_{cls_name}_median_offset"] = float("inf")
            results[f"positional_{cls_name}_within_1bp"] = 0.0
            results[f"positional_{cls_name}_within_5bp"] = 0.0

    return results


def compute_f1_at_optimal_threshold(gene_probs, gene_labels):
    """Sweep thresholds 0.1-0.9 and find optimal F1 for each class."""
    all_probs = np.concatenate(gene_probs, axis=0)
    all_labels = np.concatenate(gene_labels, axis=0)

    results = {}
    for cls_name, cls_idx in [("acceptor", 1), ("donor", 2)]:
        true_binary = (all_labels == cls_idx).astype(np.int32)
        scores = all_probs[:, cls_idx]

        best_f1 = 0.0
        best_thresh = 0.0
        for thresh in np.arange(0.1, 0.91, 0.05):
            pred = (scores >= thresh).astype(np.int32)
            tp = (pred & true_binary).sum()
            fp = (pred & ~true_binary).sum()
            fn = (~pred & true_binary).sum()

            precision = tp / max(tp + fp, 1)
            recall = tp / max(tp + fn, 1)
            f1 = 2 * precision * recall / max(precision + recall, 1e-8)

            if f1 > best_f1:
                best_f1 = f1
                best_thresh = thresh

        results[f"f1_{cls_name}_best"] = float(best_f1)
        results[f"f1_{cls_name}_threshold"] = float(best_thresh)

    return results


def compute_bootstrap_ci(gene_probs, gene_labels, n_bootstrap=1000, seed=42):
    """Bootstrap 95% confidence intervals on AUPRC by resampling genes."""
    rng = np.random.RandomState(seed)
    n_genes = len(gene_probs)

    donor_auprcs = []
    acceptor_auprcs = []

    for _ in range(n_bootstrap):
        idx = rng.choice(n_genes, size=n_genes, replace=True)
        sampled_probs = [gene_probs[i] for i in idx]
        sampled_labels = [gene_labels[i] for i in idx]

        auprc = compute_auprc(sampled_probs, sampled_labels)
        donor_auprcs.append(auprc["auprc_donor"])
        acceptor_auprcs.append(auprc["auprc_acceptor"])

    donor_arr = np.array(donor_auprcs)
    acc_arr = np.array(acceptor_auprcs)
    mean_arr = (donor_arr + acc_arr) / 2

    return {
        "ci95_auprc_donor": (float(np.percentile(donor_arr, 2.5)),
                              float(np.percentile(donor_arr, 97.5))),
        "ci95_auprc_acceptor": (float(np.percentile(acc_arr, 2.5)),
                                 float(np.percentile(acc_arr, 97.5))),
        "ci95_auprc_mean": (float(np.percentile(mean_arr, 2.5)),
                             float(np.percentile(mean_arr, 97.5))),
    }


# ---------------------------------------------------------------------------
# Threshold sweep, stratified metrics, serialization
# — keep identical to evaluate.py
# ---------------------------------------------------------------------------

def compute_threshold_sweep(gene_probs, gene_labels):
    """Full threshold sweep: precision, recall, F1 at each threshold."""
    all_probs = np.concatenate(gene_probs, axis=0)
    all_labels = np.concatenate(gene_labels, axis=0)

    results = {}
    for cls_name, cls_idx in [("acceptor", 1), ("donor", 2)]:
        true_binary = (all_labels == cls_idx).astype(np.int32)
        scores = all_probs[:, cls_idx]

        sweep = []
        for thresh in np.arange(0.05, 0.96, 0.05):
            pred = (scores >= thresh).astype(np.int32)
            tp = int((pred & true_binary).sum())
            fp = int((pred & ~true_binary).sum())
            fn = int((~pred & true_binary).sum())

            precision = tp / max(tp + fp, 1)
            recall = tp / max(tp + fn, 1)
            f1 = 2 * precision * recall / max(precision + recall, 1e-8)

            sweep.append({
                "threshold": round(float(thresh), 2),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "tp": tp, "fp": fp, "fn": fn,
            })
        results[cls_name] = sweep

    return results


def parse_gene_junctions(datafile_path):
    """Parse junction data for all test genes.

    Returns a list of dicts, one per gene, with keys:
        tx_start, tx_end, donors, acceptors,
        intron_lengths, exon_lengths
    """
    genes = []
    with h5py.File(datafile_path, "r") as f:
        n_genes = f["TX_START"].shape[0]
        for g in range(n_genes):
            tx_start = int(f["TX_START"][g])
            tx_end = int(f["TX_END"][g])

            jn_start_raw = f["JN_START"][g]
            jn_end_raw = f["JN_END"][g]
            if isinstance(jn_start_raw, np.ndarray):
                jn_start_raw = jn_start_raw[0]
            if isinstance(jn_end_raw, np.ndarray):
                jn_end_raw = jn_end_raw[0]
            if isinstance(jn_start_raw, bytes):
                jn_start_raw = jn_start_raw.decode()
            if isinstance(jn_end_raw, bytes):
                jn_end_raw = jn_end_raw.decode()

            donors = sorted([int(x) for x in jn_start_raw.split(",") if x.strip()])
            acceptors = sorted([int(x) for x in jn_end_raw.split(",") if x.strip()])

            intron_lengths = {}
            for d, a in zip(donors, acceptors):
                intron_len = a - d
                intron_lengths[d] = intron_len
                intron_lengths[a] = intron_len

            exon_lengths = {}
            if len(donors) > 1:
                for i in range(len(acceptors) - 1):
                    exon_len = donors[i + 1] - acceptors[i]
                    exon_lengths[acceptors[i]] = exon_len
                    exon_lengths[donors[i + 1]] = exon_len

            if donors:
                first_exon = donors[0] - tx_start
                exon_lengths[donors[0]] = first_exon
            if acceptors:
                last_exon = tx_end - acceptors[-1]
                exon_lengths[acceptors[-1]] = last_exon

            genes.append({
                "tx_start": tx_start,
                "tx_end": tx_end,
                "donors": donors,
                "acceptors": acceptors,
                "intron_lengths": intron_lengths,
                "exon_lengths": exon_lengths,
            })

    return genes


def compute_stratified_metrics(gene_probs, gene_labels, genes):
    """Compute recall stratified by intron and exon length buckets."""
    intron_buckets = {
        "<200bp": (0, 200),
        "200-1000bp": (200, 1000),
        "1000-5000bp": (1000, 5000),
        ">5000bp": (5000, float("inf")),
    }
    exon_buckets = {
        "<80bp": (0, 80),
        "80-200bp": (80, 200),
        "200-500bp": (200, 500),
        ">500bp": (500, float("inf")),
    }

    bucket_scores = defaultdict(list)

    for probs, labels, gene_info in zip(gene_probs, gene_labels, genes):
        tx_start = gene_info["tx_start"]

        for cls_name, cls_idx in [("acceptor", 1), ("donor", 2)]:
            scores = probs[:, cls_idx]
            true_positions = np.where(labels == cls_idx)[0]

            for pos in true_positions:
                genomic_coord = pos + tx_start

                intron_len = gene_info["intron_lengths"].get(genomic_coord)
                if intron_len is not None:
                    for bname, (lo, hi) in intron_buckets.items():
                        if lo <= intron_len < hi:
                            bucket_scores[("intron", bname, cls_name)].append(
                                float(scores[pos])
                            )
                            break

                exon_len = gene_info["exon_lengths"].get(genomic_coord)
                if exon_len is not None:
                    for bname, (lo, hi) in exon_buckets.items():
                        if lo <= exon_len < hi:
                            bucket_scores[("exon", bname, cls_name)].append(
                                float(scores[pos])
                            )
                            break

    results = {"by_intron_length": {}, "by_exon_length": {}}
    for (strat_type, bname, cls_name), scores_list in bucket_scores.items():
        scores_arr = np.array(scores_list)
        n_sites = len(scores_arr)
        key = f"by_{strat_type}_length"

        if bname not in results[key]:
            results[key][bname] = {}

        results[key][bname][f"{cls_name}_n_sites"] = n_sites
        results[key][bname][f"{cls_name}_mean_score"] = float(scores_arr.mean())
        results[key][bname][f"{cls_name}_median_score"] = float(np.median(scores_arr))

        for thresh in [0.3, 0.5, 0.7]:
            recall = float((scores_arr >= thresh).mean())
            results[key][bname][f"{cls_name}_recall_at_{thresh}"] = recall

    return results


def make_serializable(obj):
    """Convert numpy types and handle inf/nan for JSON serialization."""
    if isinstance(obj, (np.floating, np.integer)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, float) and (math.isinf(obj) or math.isnan(obj)):
        return str(obj)
    if isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [make_serializable(v) for v in obj]
    if isinstance(obj, tuple):
        return [make_serializable(v) for v in obj]
    return obj


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(output_dir="results"):
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
        },
    }

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    json_path = out_path / "spliceai_results.json"
    with open(json_path, "w") as fp:
        json.dump(make_serializable(all_results), fp, indent=2)
    print(f"\nResults saved to {json_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate SpliceAI baseline")
    parser.add_argument("--output-dir", type=str, default="results",
                        help="Directory to save results JSON (default: results/)")
    args = parser.parse_args()
    main(output_dir=args.output_dir)
