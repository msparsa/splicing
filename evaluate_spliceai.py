"""
Evaluate pre-trained SpliceAI (10k context, 5-model ensemble) on the test set.

Metrics are computed identically to evaluate.py so results are directly
comparable with SpliceMamba.

Usage:
    /mnt/lareaulab/mparsa/miniconda3/envs/spliceai_env/bin/python evaluate_spliceai.py
"""

from __future__ import annotations

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['PYTHONUNBUFFERED'] = '1'

import sys
import time

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

def load_spliceai_model():
    """Load a single pre-trained SpliceAI 10k model from the pip package."""
    from keras.models import load_model

    path = resource_filename("spliceai", "models/spliceai1.h5")
    print(f"  Loading {path} ...")
    model = load_model(path, compile=False)
    return model


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def predict_windows(model, dataset_path, cfg):
    """Run single-model inference on all test windows.

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

            preds = model.predict(x_data, batch_size=cfg["batch_size"],
                                  verbose=0)
            # preds shape: (N, 5000, 3) — softmax probabilities
            all_probs.append(preds)

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
# Main
# ---------------------------------------------------------------------------

def main():
    cfg = CONFIG
    start_time = time.time()

    # Load model
    print("Loading SpliceAI 10k model (single model)...")
    model = load_spliceai_model()
    print("  Loaded 1 model")

    # Run inference
    print("\nRunning inference on test set...")
    all_probs = predict_windows(model, cfg["test_dataset_path"], cfg)
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
    print("EVALUATION RESULTS — SpliceAI 10k (single model)")
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

    elapsed = time.time() - start_time
    print(f"\n{'=' * 60}")
    print(f"Total time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
