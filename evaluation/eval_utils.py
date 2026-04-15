"""
Shared evaluation utilities for splice site prediction models.

Contains metric functions, gene-level stitching, and label processing
used by evaluate.py, evaluate_spliceai.py, evaluate_pangolin.py,
evaluate_tissue.py, and evaluate_psi.py.
"""

from __future__ import annotations

import math
from collections import defaultdict

import h5py
import numpy as np
from scipy.signal import find_peaks
from sklearn.metrics import average_precision_score, roc_curve, auc


# ---------------------------------------------------------------------------
# Gene-level stitching & label I/O
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
    """Stitch window predictions into gene-level arrays.

    For 3-class models: all_probs shape (total_windows, 5000, 3)
        -> list of (n_win*5000, 3) arrays
    For binary models: all_probs shape (total_windows, 5000)
        -> list of (n_win*5000,) arrays
    """
    gene_probs = []
    offset = 0
    ndim = all_probs.ndim
    for n_win in windows_per_gene:
        n_win = int(n_win)
        gene_pred = all_probs[offset : offset + n_win]
        if ndim == 3:
            gene_pred = gene_pred.reshape(-1, all_probs.shape[-1])
        else:
            gene_pred = gene_pred.reshape(-1)
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
    return np.concatenate(all_labels, axis=0)


def stitch_gene_labels(
    all_labels: np.ndarray,
    windows_per_gene: np.ndarray,
) -> list[np.ndarray]:
    """Stitch window-level labels into gene-level label arrays."""
    gene_labels = []
    offset = 0
    for n_win in windows_per_gene:
        n_win = int(n_win)
        gene_lab = all_labels[offset : offset + n_win]
        gene_lab = gene_lab.reshape(-1)
        gene_labels.append(gene_lab)
        offset += n_win
    return gene_labels


# ---------------------------------------------------------------------------
# Binary splice conversion (for Pangolin comparison)
# ---------------------------------------------------------------------------

def adapt_to_binary_splice(gene_probs_3class: list[np.ndarray]) -> list[np.ndarray]:
    """Convert 3-class probs to single splice probability.

    P(splice) = P(acceptor) + P(donor) = probs[:, 1] + probs[:, 2]
    """
    return [p[:, 1] + p[:, 2] for p in gene_probs_3class]


def labels_to_binary(gene_labels: list[np.ndarray]) -> list[np.ndarray]:
    """Convert 3-class labels {0=neither, 1=acceptor, 2=donor} to binary {0, 1}."""
    return [(lab > 0).astype(np.int32) for lab in gene_labels]


# ---------------------------------------------------------------------------
# 3-class metrics (donor/acceptor-specific)
# ---------------------------------------------------------------------------

def compute_auprc(gene_probs: list[np.ndarray], gene_labels: list[np.ndarray]) -> dict:
    """Compute AUPRC for donor and acceptor classes across all genes."""
    all_probs = np.concatenate(gene_probs, axis=0)
    all_labels = np.concatenate(gene_labels, axis=0)

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


def compute_roc(gene_probs: list[np.ndarray], gene_labels: list[np.ndarray]) -> dict:
    """Compute ROC curve and AUROC for donor and acceptor classes."""
    all_probs = np.concatenate(gene_probs, axis=0)
    all_labels = np.concatenate(gene_labels, axis=0)

    results = {}
    for cls_name, cls_idx in [("acceptor", 1), ("donor", 2)]:
        true_binary = (all_labels == cls_idx).astype(np.int32)
        scores = all_probs[:, cls_idx]
        fpr, tpr, thresholds = roc_curve(true_binary, scores)
        auroc = auc(fpr, tpr)
        results[f"roc_{cls_name}"] = {"fpr": fpr, "tpr": tpr, "thresholds": thresholds}
        results[f"auroc_{cls_name}"] = auroc

    results["auroc_mean"] = (results["auroc_donor"] + results["auroc_acceptor"]) / 2.0
    return results


def compute_topk_accuracy(
    gene_probs: list[np.ndarray],
    gene_labels: list[np.ndarray],
) -> dict:
    """Compute top-k accuracy both globally (SpliceAI method) and per-gene."""
    all_probs = np.concatenate(gene_probs, axis=0)
    all_labels = np.concatenate(gene_labels, axis=0)

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


def compute_f1_at_optimal_threshold(
    gene_probs: list[np.ndarray],
    gene_labels: list[np.ndarray],
) -> dict:
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


def compute_bootstrap_ci(
    gene_probs: list[np.ndarray],
    gene_labels: list[np.ndarray],
    n_bootstrap: int = 200,
    seed: int = 42,
) -> dict:
    """Bootstrap 95% confidence intervals on AUPRC by resampling genes.

    Pre-computes per-gene AUPRC to avoid repeated concatenation of all
    positions (~82M) on every iteration.
    """
    rng = np.random.RandomState(seed)
    n_genes = len(gene_probs)

    # Pre-compute per-gene AUPRC for donor and acceptor
    pg_donor = np.zeros(n_genes)
    pg_acceptor = np.zeros(n_genes)
    pg_weight = np.zeros(n_genes)
    for i, (p, l) in enumerate(zip(gene_probs, gene_labels)):
        n_classes = p.shape[-1]
        labels_flat = l.ravel()
        pg_weight[i] = len(labels_flat)
        # Donor
        donor_true = (labels_flat == 2).astype(np.int32)
        if donor_true.sum() > 0:
            pg_donor[i] = average_precision_score(donor_true, p[:, 2] if n_classes > 2 else p.ravel())
        # Acceptor
        acc_true = (labels_flat == 1).astype(np.int32)
        if acc_true.sum() > 0:
            pg_acceptor[i] = average_precision_score(acc_true, p[:, 1] if n_classes > 1 else p.ravel())
    pg_weight /= pg_weight.sum()

    donor_auprcs = []
    acceptor_auprcs = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n_genes, size=n_genes, replace=True)
        w = pg_weight[idx]; w = w / w.sum()
        donor_auprcs.append(float(np.average(pg_donor[idx], weights=w)))
        acceptor_auprcs.append(float(np.average(pg_acceptor[idx], weights=w)))

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
        thresholds = list(np.arange(0.05, 0.96, 0.05)) + [0.95, 0.96, 0.97, 0.98, 0.99, 1.00]
        thresholds = sorted(set(round(t, 2) for t in thresholds))
        for thresh in thresholds:
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


# ---------------------------------------------------------------------------
# Binary splice metrics (for Pangolin / tissue-specific comparison)
# ---------------------------------------------------------------------------

def compute_binary_auprc(
    gene_probs_splice: list[np.ndarray],
    gene_labels_binary: list[np.ndarray],
) -> dict:
    """AUPRC for binary splice-vs-not-splice classification."""
    all_probs = np.concatenate(gene_probs_splice, axis=0)
    all_labels = np.concatenate(gene_labels_binary, axis=0)
    auprc = average_precision_score(all_labels, all_probs)
    return {"auprc_splice": float(auprc)}


def compute_binary_topk(
    gene_probs_splice: list[np.ndarray],
    gene_labels_binary: list[np.ndarray],
) -> dict:
    """Top-k accuracy for binary splice classification.

    Global: k = total number of true splice sites.
    Per-gene: k = {0.5, 1, 2, 4} * n_true_per_gene.
    """
    all_probs = np.concatenate(gene_probs_splice, axis=0)
    all_labels = np.concatenate(gene_labels_binary, axis=0)

    true_mask = all_labels > 0
    n_true = int(true_mask.sum())
    results = {}
    if n_true > 0:
        top_idx = np.argpartition(-all_probs, n_true)[:n_true]
        n_found = true_mask[top_idx].sum()
        results["topk_global_splice"] = float(n_found) / float(n_true)
    else:
        results["topk_global_splice"] = 0.0

    per_gene = {f"topk_splice_k{k}": [] for k in [0.5, 1, 2, 4]}
    for probs, labels in zip(gene_probs_splice, gene_labels_binary):
        true_mask = labels > 0
        n_true = true_mask.sum()
        if n_true == 0:
            continue
        for k in [0.5, 1, 2, 4]:
            n_select = max(1, int(k * n_true))
            top_idx = np.argpartition(-probs, n_select)[:n_select]
            n_found = true_mask[top_idx].sum()
            per_gene[f"topk_splice_k{k}"].append(float(n_found) / float(n_true))

    for k, v in per_gene.items():
        results[k] = float(np.mean(v)) if v else 0.0

    return results


def compute_binary_f1(
    gene_probs_splice: list[np.ndarray],
    gene_labels_binary: list[np.ndarray],
) -> dict:
    """F1 at optimal threshold for binary splice classification."""
    all_probs = np.concatenate(gene_probs_splice, axis=0)
    all_labels = np.concatenate(gene_labels_binary, axis=0).astype(np.int32)

    best_f1 = 0.0
    best_thresh = 0.0
    for thresh in np.arange(0.05, 0.96, 0.05):
        pred = (all_probs >= thresh).astype(np.int32)
        tp = (pred & all_labels).sum()
        fp = (pred & ~all_labels).sum()
        fn = (~pred & all_labels).sum()
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh

    return {"f1_splice_best": float(best_f1), "f1_splice_threshold": float(best_thresh)}


def compute_binary_positional(
    gene_probs_splice: list[np.ndarray],
    gene_labels_binary: list[np.ndarray],
    peak_height: float = 0.3,
    peak_distance: int = 20,
) -> dict:
    """Positional accuracy for binary splice predictions."""
    offsets = []
    for probs, labels in zip(gene_probs_splice, gene_labels_binary):
        true_positions = np.where(labels > 0)[0]
        if len(true_positions) == 0:
            continue
        peaks, _ = find_peaks(probs, height=peak_height, distance=peak_distance)
        if len(peaks) == 0:
            continue
        for tp in true_positions:
            dists = np.abs(peaks.astype(np.int64) - tp)
            offsets.append(int(dists.min()))

    if offsets:
        arr = np.array(offsets)
        return {
            "positional_splice_mean_offset": float(arr.mean()),
            "positional_splice_median_offset": float(np.median(arr)),
            "positional_splice_within_1bp": float((arr <= 1).mean()),
            "positional_splice_within_5bp": float((arr <= 5).mean()),
        }
    return {
        "positional_splice_mean_offset": float("inf"),
        "positional_splice_median_offset": float("inf"),
        "positional_splice_within_1bp": 0.0,
        "positional_splice_within_5bp": 0.0,
    }


def compute_binary_bootstrap_ci(
    gene_probs_splice: list[np.ndarray],
    gene_labels_binary: list[np.ndarray],
    n_bootstrap: int = 200,
    seed: int = 42,
) -> dict:
    """Bootstrap 95% CI on binary splice AUPRC (gene-level resampling).

    Uses pre-computed per-gene AUPRC to avoid repeated concatenation of
    ~82M positions, making bootstrap ~1000x faster.
    """
    rng = np.random.RandomState(seed)
    n_genes = len(gene_probs_splice)

    # Pre-compute per-gene AUPRC (only genes with at least 1 splice site)
    per_gene_auprc = []
    per_gene_weight = []  # weight by number of positions
    for p, l in zip(gene_probs_splice, gene_labels_binary):
        if l.sum() > 0:
            per_gene_auprc.append(average_precision_score(l, p))
            per_gene_weight.append(len(l))
        else:
            per_gene_auprc.append(0.0)
            per_gene_weight.append(len(l))
    per_gene_auprc = np.array(per_gene_auprc)
    per_gene_weight = np.array(per_gene_weight, dtype=np.float64)
    per_gene_weight /= per_gene_weight.sum()

    auprcs = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n_genes, size=n_genes, replace=True)
        w = per_gene_weight[idx]
        w = w / w.sum()
        auprcs.append(float(np.average(per_gene_auprc[idx], weights=w)))
    arr = np.array(auprcs)
    return {
        "ci95_auprc_splice": (float(np.percentile(arr, 2.5)),
                               float(np.percentile(arr, 97.5))),
    }


# ---------------------------------------------------------------------------
# Gene junction parsing & stratified metrics
# ---------------------------------------------------------------------------

def parse_gene_junctions(datafile_path: str) -> list[dict]:
    """Parse junction and sequence data for all test genes."""
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
                            bucket_scores[("intron", bname, cls_name)].append(float(scores[pos]))
                            break
                exon_len = gene_info["exon_lengths"].get(genomic_coord)
                if exon_len is not None:
                    for bname, (lo, hi) in exon_buckets.items():
                        if lo <= exon_len < hi:
                            bucket_scores[("exon", bname, cls_name)].append(float(scores[pos]))
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


# ---------------------------------------------------------------------------
# JSON serialization
# ---------------------------------------------------------------------------

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
