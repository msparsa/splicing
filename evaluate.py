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
import json
import math
from collections import defaultdict
from datetime import datetime
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


def compute_threshold_sweep(
    gene_probs: list[np.ndarray],
    gene_labels: list[np.ndarray],
) -> dict:
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


def parse_gene_junctions(datafile_path: str) -> list[dict]:
    """Parse junction and sequence data for all test genes.

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

            # Parse junction starts (donors) and ends (acceptors)
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

            # Compute intron lengths
            intron_lengths = {}
            for d, a in zip(donors, acceptors):
                intron_len = a - d
                intron_lengths[d] = intron_len  # keyed by donor position
                intron_lengths[a] = intron_len  # also keyed by acceptor position

            # Compute exon lengths (internal exons between consecutive junctions)
            exon_lengths = {}
            if len(donors) > 1:
                for i in range(len(acceptors) - 1):
                    exon_len = donors[i + 1] - acceptors[i]
                    exon_lengths[acceptors[i]] = exon_len
                    exon_lengths[donors[i + 1]] = exon_len

            # First exon (before first donor)
            if donors:
                first_exon = donors[0] - tx_start
                exon_lengths[donors[0]] = first_exon
            # Last exon (after last acceptor)
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


def compute_stratified_metrics(
    gene_probs: list[np.ndarray],
    gene_labels: list[np.ndarray],
    genes: list[dict],
) -> dict:
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

    # Collect predicted scores for true splice sites, keyed by bucket
    bucket_scores = defaultdict(list)

    for probs, labels, gene_info in zip(gene_probs, gene_labels, genes):
        tx_start = gene_info["tx_start"]

        for cls_name, cls_idx in [("acceptor", 1), ("donor", 2)]:
            scores = probs[:, cls_idx]
            true_positions = np.where(labels == cls_idx)[0]

            for pos in true_positions:
                genomic_coord = pos + tx_start

                # Intron length bucketing
                intron_len = gene_info["intron_lengths"].get(genomic_coord)
                if intron_len is not None:
                    for bname, (lo, hi) in intron_buckets.items():
                        if lo <= intron_len < hi:
                            bucket_scores[("intron", bname, cls_name)].append(
                                float(scores[pos])
                            )
                            break

                # Exon length bucketing
                exon_len = gene_info["exon_lengths"].get(genomic_coord)
                if exon_len is not None:
                    for bname, (lo, hi) in exon_buckets.items():
                        if lo <= exon_len < hi:
                            bucket_scores[("exon", bname, cls_name)].append(
                                float(scores[pos])
                            )
                            break

    # Compute per-bucket metrics
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
# Main evaluation
# ---------------------------------------------------------------------------

def evaluate(checkpoint_path: str, cfg: dict, output_dir: str = "results"):
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

    print("\n" + "=" * 60)

    # Save results to JSON
    all_results = {
        "model": "splicemamba",
        "checkpoint": str(checkpoint_path),
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
    parser.add_argument("--output-dir", type=str, default="results",
                        help="Directory to save results JSON (default: results/)")
    args = parser.parse_args()
    evaluate(args.checkpoint, EVAL_CONFIG, output_dir=args.output_dir)
