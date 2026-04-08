"""
SpliceMamba evaluation script.

Runs inference on the test set (chr1, 1,652 genes), performs gene-level
stitching, and computes all metrics from SPEC.md Section 7.

Usage:
    python evaluate.py --checkpoint checkpoints/best.pt
    python evaluate.py --checkpoint checkpoints/last.pt
"""

from __future__ import annotations


import argparse
from pathlib import Path

import h5py
import numpy as np
import torch
from scipy.signal import find_peaks
from sklearn.metrics import average_precision_score

from model import SpliceMamba

# ---------------------------------------------------------------------------
# Configuration (must match training)
# ---------------------------------------------------------------------------

EVAL_CONFIG = dict(
    test_dataset_path="dataset_test_0.h5",
    test_datafile_path="datafile_test_0.h5",
    d_model=256,
    n_mamba_layers=8,
    d_state=64,
    expand=2,
    d_conv=4,
    headdim=32,
    n_attn_layers=4,
    n_heads=8,
    window_radius=400,
    dropout=0.1,
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
    model = SpliceMamba(
        d_model=cfg["d_model"],
        n_mamba_layers=cfg["n_mamba_layers"],
        d_state=cfg["d_state"],
        expand=cfg["expand"],
        d_conv=cfg["d_conv"],
        headdim=cfg["headdim"],
        n_attn_layers=cfg["n_attn_layers"],
        n_heads=cfg["n_heads"],
        window_radius=cfg["window_radius"],
        dropout=cfg["dropout"],
        n_classes=cfg["n_classes"],
        max_len=cfg["max_len"],
    ).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Loaded checkpoint from epoch {ckpt.get('epoch', '?')}, "
          f"best AUPRC: {ckpt.get('best_auprc', '?')}")
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

                # Softmax on label region
                label_logits = refined_logits[
                    :, cfg["label_start"]:cfg["label_end"], :
                ]
                probs = torch.softmax(label_logits.float(), dim=-1)
                shard_probs.append(probs.cpu().numpy())

            all_probs.append(np.concatenate(shard_probs, axis=0))

    return np.concatenate(all_probs, axis=0)  # (total_windows, 5000, 3)


# ---------------------------------------------------------------------------
# Gene-level stitching
# ---------------------------------------------------------------------------

def compute_gene_window_counts(datafile_path: str) -> np.ndarray:
    """Return per-gene window counts: ceil(gene_len / 5000)."""
    with h5py.File(datafile_path, "r") as f:
        tx_start = f["TX_START"][:]
        tx_end = f["TX_END"][:]
    gene_lens = tx_end - tx_start
    return np.ceil(gene_lens / 5000).astype(np.int64)


def stitch_gene_predictions(
    all_probs: np.ndarray,
    windows_per_gene: np.ndarray,
) -> list[np.ndarray]:
    """Stitch overlapping window predictions into gene-level probability arrays.

    Each gene's windows cover non-overlapping 5kb label regions
    (SpliceAI windowing). Adjacent windows' label regions tile the gene
    without overlap, so stitching is simple concatenation.

    Returns a list of (gene_len_labels, 3) arrays.
    """
    gene_probs = []
    offset = 0
    for n_win in windows_per_gene:
        n_win = int(n_win)
        # Each window contributes 5000 label positions
        gene_pred = all_probs[offset : offset + n_win]  # (n_win, 5000, 3)
        # Concatenate along position axis
        gene_pred = gene_pred.reshape(-1, 3)  # (n_win * 5000, 3)
        gene_probs.append(gene_pred)
        offset += n_win
    return gene_probs


def read_window_labels(dataset_path: str) -> np.ndarray:
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


def stitch_gene_labels(
    all_labels: np.ndarray,
    windows_per_gene: np.ndarray,
) -> list[np.ndarray]:
    """Stitch window-level labels into gene-level label arrays.

    Returns a list of 1-D int64 arrays (one per gene), each of length
    n_windows * 5000.
    """
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
# Metrics
# ---------------------------------------------------------------------------

def compute_auprc(gene_probs: list[np.ndarray], gene_labels: list[np.ndarray]) -> dict:
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


def compute_topk_accuracy(
    gene_probs: list[np.ndarray],
    gene_labels: list[np.ndarray],
) -> dict:
    """Compute top-k accuracy both globally (SpliceAI method) and per-gene.

    Global: pool all positions across all genes, pick top-k globally where
    k = total number of true sites for that class. This matches SpliceAI's
    reported metric.

    Per-gene: compute per gene then average (our original method).
    """
    # --- Global top-k (SpliceAI method) ---
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


def compute_positional_accuracy(
    gene_probs: list[np.ndarray],
    gene_labels: list[np.ndarray],
    peak_height: float = 0.5,
    peak_distance: int = 20,
) -> dict:
    """Compute positional accuracy: distribution of offsets between predicted
    peaks and nearest true splice sites."""
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

            # For each true site, find nearest predicted peak
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


def compute_f1_at_optimal_threshold(
    gene_probs: list[np.ndarray],
    gene_labels: list[np.ndarray],
) -> dict:
    """Sweep thresholds 0.1–0.9 and find optimal F1 for each class."""
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


def compute_bootstrap_ci(
    gene_probs: list[np.ndarray],
    gene_labels: list[np.ndarray],
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> dict:
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
# Main evaluation
# ---------------------------------------------------------------------------

def evaluate(checkpoint_path: str, cfg: dict):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = load_model(checkpoint_path, cfg, device)

    # Run inference on all test windows
    print("Running inference on test set...")
    all_probs = predict_windows(model, cfg["test_dataset_path"], cfg, device)
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

    print("\n" + "=" * 60)

    # Return all metrics as a dict
    return {**auprc, **topk, **pos, **f1}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate SpliceMamba")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    args = parser.parse_args()
    evaluate(args.checkpoint, EVAL_CONFIG)
