"""
Diagnostic experiments comparing SpliceMamba and SpliceAI behavior on chr1.

Produces:
  * Experiment 1 — Per-class probability scatter plots (both models)
  * Experiment 2a — Performance vs distance-to-gene-edge (position in context)
  * Experiment 3 — Error-overlap Venn/bar per class
  * Experiment 4 — Calibration/reliability diagrams

Experiment 2b (in-silico context masking) requires fresh inference and lives
in ``context_masking.py`` + ``plot_context_masking.py``.

Usage:
    python evaluation/analyze_behavior.py --experiment all
    python evaluation/analyze_behavior.py --experiment scatter
    python evaluation/analyze_behavior.py --experiment position
    python evaluation/analyze_behavior.py --experiment overlap
    python evaluation/analyze_behavior.py --experiment calibration
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss

_REPO_ROOT = Path(__file__).resolve().parent.parent
_EVAL_DIR = _REPO_ROOT / "evaluation"
if str(_EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(_EVAL_DIR))

from eval_utils import (  # noqa: E402
    compute_gene_window_counts,
    read_window_labels,
    stitch_gene_labels,
    stitch_gene_predictions,
    make_serializable,
)

# ---------------------------------------------------------------------------
# Paths / config
# ---------------------------------------------------------------------------

SPLICEMAMBA_NPZ = _EVAL_DIR / "results" / "splicemamba_ensemble_preds.npz"
SPLICEAI_NPZ = _EVAL_DIR / "results" / "spliceai_preds.npz"
DATASET_H5 = _REPO_ROOT / "dataset_test_0.h5"
DATAFILE_H5 = _REPO_ROOT / "datafile_test_0.h5"
OUT_DIR = _EVAL_DIR / "results" / "behavior"

CLASS_NAMES = {0: "neither", 1: "acceptor", 2: "donor"}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_predictions() -> dict:
    """Load cached predictions and labels, return flat arrays.

    Returns a dict with:
        sm_probs, sa_probs: (N, 3) float32 — flat per-position softmax for each model
        labels:             (N,)   int64   — per-position class index
        sm_rescale:         dict with per-class (p_min, p_max) used for rescaling
        sm_probs_rescaled:  (N, 3) float32 — SpliceMamba probs linearly rescaled
                            per-class to [0, 1] using empirical min/max
    """
    print("[load] reading predictions …", file=sys.stderr)
    sm_raw = np.load(SPLICEMAMBA_NPZ)["probs"]
    sa_raw = np.load(SPLICEAI_NPZ)["probs"]
    assert sm_raw.shape == sa_raw.shape, (sm_raw.shape, sa_raw.shape)

    sm_probs = sm_raw.reshape(-1, sm_raw.shape[-1])
    sa_probs = sa_raw.reshape(-1, sa_raw.shape[-1])

    print("[load] reading labels …", file=sys.stderr)
    labels = read_window_labels(str(DATASET_H5)).reshape(-1)
    assert labels.shape[0] == sm_probs.shape[0], (labels.shape, sm_probs.shape)

    print(f"[load] total positions: {labels.shape[0]:,}", file=sys.stderr)
    print(
        f"[load]   class counts — neither={int((labels == 0).sum()):,} "
        f"acceptor={int((labels == 1).sum()):,} donor={int((labels == 2).sum()):,}",
        file=sys.stderr,
    )

    # Per-class empirical min/max for SpliceMamba rescaling.
    sm_rescale = {}
    sm_probs_rescaled = np.zeros_like(sm_probs)
    print("[load] SpliceMamba raw per-class probability bounds:", file=sys.stderr)
    for cls_idx, cls_name in CLASS_NAMES.items():
        col = sm_probs[:, cls_idx]
        p_min = float(col.min())
        p_max = float(col.max())
        p_mean = float(col.mean())
        sm_rescale[cls_name] = {"min": p_min, "max": p_max, "mean": p_mean}
        print(
            f"[load]   {cls_name}: min={p_min:.4f}  max={p_max:.4f}  mean={p_mean:.4f}",
            file=sys.stderr,
        )
        if p_max > p_min:
            sm_probs_rescaled[:, cls_idx] = np.clip(
                (col - p_min) / (p_max - p_min), 0.0, 1.0
            )
        else:
            sm_probs_rescaled[:, cls_idx] = 0.0

    return {
        "sm_probs": sm_probs,
        "sm_probs_rescaled": sm_probs_rescaled.astype(np.float32),
        "sa_probs": sa_probs,
        "labels": labels,
        "sm_rescale": sm_rescale,
    }


# ---------------------------------------------------------------------------
# Experiment 1: per-class probability scatter
# ---------------------------------------------------------------------------

def experiment_scatter(data: dict, out_dir: Path) -> dict:
    """Scatter plot (or hexbin for 'neither') of SpliceAI vs SpliceMamba per-class
    probability, limited to positions whose true class matches the plot class."""
    print("[exp1] probability scatter plots …", file=sys.stderr)
    sm = data["sm_probs_rescaled"]
    sa = data["sa_probs"]
    labels = data["labels"]

    THRESH = 0.5
    summary = {}

    for cls_idx, cls_name in CLASS_NAMES.items():
        mask = labels == cls_idx
        n = int(mask.sum())
        x = sa[mask, cls_idx]
        y = sm[mask, cls_idx]

        fig, ax = plt.subplots(figsize=(6.5, 6.5))

        if cls_name == "neither":
            hb = ax.hexbin(x, y, gridsize=60, bins="log", cmap="viridis", mincnt=1)
            cbar = fig.colorbar(hb, ax=ax)
            cbar.set_label("log10(count)")
        else:
            # Color by correctness pattern
            both_right = (x >= THRESH) & (y >= THRESH)
            both_wrong = (x < THRESH) & (y < THRESH)
            only_sa = (x >= THRESH) & (y < THRESH)
            only_sm = (x < THRESH) & (y >= THRESH)
            for m, colour, lbl in [
                (both_right, "tab:green", f"both ≥ {THRESH} (n={int(both_right.sum())})"),
                (both_wrong, "tab:red", f"both < {THRESH} (n={int(both_wrong.sum())})"),
                (only_sa, "tab:orange", f"only SpliceAI (n={int(only_sa.sum())})"),
                (only_sm, "tab:blue", f"only SpliceMamba (n={int(only_sm.sum())})"),
            ]:
                ax.scatter(x[m], y[m], s=6, alpha=0.35, c=colour, label=lbl,
                           edgecolors="none")
            ax.legend(loc="lower right", fontsize=8, framealpha=0.9)

        ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.6)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel(f"SpliceAI  P({cls_name})")
        ax.set_ylabel(f"SpliceMamba  P({cls_name})  [rescaled]")
        corr = float(np.corrcoef(x, y)[0, 1]) if n > 1 else 0.0
        ax.set_title(
            f"True {cls_name} positions  (n={n:,},  Pearson r={corr:.3f})"
        )
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        out_path = out_dir / f"prob_scatter_{cls_name}.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"[exp1]   wrote {out_path.name}", file=sys.stderr)

        summary[cls_name] = {"n": n, "pearson_r": corr}

    return summary


# ---------------------------------------------------------------------------
# Experiment 2a: position-within-context
# ---------------------------------------------------------------------------

DIST_BUCKETS = [
    ("0-500bp", 0, 500),
    ("500-2500bp", 500, 2500),
    ("2500-5000bp", 2500, 5000),
    ("5000-7500bp", 5000, 7500),
    (">7500bp", 7500, float("inf")),
]


def _compute_dist_to_edge(n_genes: int, gene_lens: np.ndarray) -> list[np.ndarray]:
    """For each gene, return an array of dist_to_nearest_boundary values with
    shape matching the stitched label array (n_windows * 5000 per gene).

    The stitched array length may exceed gene_len (right-padding from ceil);
    for those padding positions, dist is negative (they are out-of-gene)."""
    dists = []
    for gi in range(n_genes):
        gl = int(gene_lens[gi])
        n_win = int(np.ceil(gl / 5000))
        stitched_len = n_win * 5000
        pos = np.arange(stitched_len)
        d = np.minimum(pos, gl - 1 - pos)  # dist to nearest of {0, gl-1}
        # pos >= gl are padding — mark with a very negative sentinel so they
        # won't land in any bucket.
        d[pos >= gl] = -1
        dists.append(d)
    return dists


def experiment_position(data: dict, out_dir: Path) -> dict:
    """Stratify model performance by distance-to-gene-edge."""
    print("[exp2a] position-within-context …", file=sys.stderr)
    labels_flat = data["labels"]
    sm = data["sm_probs_rescaled"]
    sa = data["sa_probs"]

    windows_per_gene = compute_gene_window_counts(str(DATAFILE_H5))

    with h5py.File(DATAFILE_H5, "r") as f:
        tx_start = f["TX_START"][:]
        tx_end = f["TX_END"][:]
    gene_lens = (tx_end - tx_start).astype(np.int64)

    # Build per-gene dist arrays then concat (so it aligns with flat labels).
    dist_per_gene = _compute_dist_to_edge(len(gene_lens), gene_lens)
    # Sanity: sum of lengths == labels_flat length
    total = sum(len(d) for d in dist_per_gene)
    assert total == labels_flat.shape[0], (total, labels_flat.shape[0])
    dist_flat = np.concatenate(dist_per_gene).astype(np.int64)

    bin_idx = np.full(dist_flat.shape, -1, dtype=np.int8)
    for bi, (_, lo, hi) in enumerate(DIST_BUCKETS):
        m = (dist_flat >= lo) & (dist_flat < hi)
        bin_idx[m] = bi

    summary = {}
    THRESH = 0.5
    # Build table: per (model, class, bucket) recall@0.5 and AUPRC
    for cls_idx, cls_name in [(1, "acceptor"), (2, "donor")]:
        cls_summary = {}
        for bi, (bname, _, _) in enumerate(DIST_BUCKETS):
            bmask = bin_idx == bi
            if not bmask.any():
                cls_summary[bname] = None
                continue
            true_mask = labels_flat[bmask] == cls_idx
            n_true = int(true_mask.sum())
            row = {"n_sites": n_true, "n_positions": int(bmask.sum())}
            if n_true == 0:
                row.update({
                    "sm_recall@0.5": None, "sa_recall@0.5": None,
                    "sm_auprc": None, "sa_auprc": None,
                })
            else:
                for key, probs in [("sm", sm), ("sa", sa)]:
                    scores = probs[bmask, cls_idx]
                    row[f"{key}_recall@0.5"] = float((scores[true_mask] >= THRESH).mean())
                    if true_mask.any() and (~true_mask).any():
                        row[f"{key}_auprc"] = float(
                            average_precision_score(true_mask.astype(int), scores)
                        )
                    else:
                        row[f"{key}_auprc"] = None
            cls_summary[bname] = row
        summary[cls_name] = cls_summary

    # ----- Plot: grouped bars for recall@0.5 per bucket, one subplot per class.
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)
    bucket_names = [b[0] for b in DIST_BUCKETS]
    x_pos = np.arange(len(bucket_names))
    width = 0.38
    for ax, (cls_idx, cls_name) in zip(axes, [(1, "acceptor"), (2, "donor")]):
        sa_vals = []
        sm_vals = []
        counts = []
        for bname in bucket_names:
            row = summary[cls_name][bname]
            if row is None or row.get("sa_recall@0.5") is None:
                sa_vals.append(0.0); sm_vals.append(0.0); counts.append(0)
            else:
                sa_vals.append(row["sa_recall@0.5"])
                sm_vals.append(row["sm_recall@0.5"])
                counts.append(row["n_sites"])
        ax.bar(x_pos - width / 2, sa_vals, width, label="SpliceAI", color="tab:orange")
        ax.bar(x_pos + width / 2, sm_vals, width, label="SpliceMamba", color="tab:blue")
        for i, c in enumerate(counts):
            ax.annotate(f"n={c}", xy=(x_pos[i], 0.02), ha="center", fontsize=8,
                        color="gray")
        ax.set_xticks(x_pos)
        ax.set_xticklabels(bucket_names, rotation=25, ha="right")
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Recall @ P ≥ 0.5")
        ax.set_title(f"{cls_name.capitalize()} recall vs distance-to-gene-edge")
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend(loc="lower right")
    fig.suptitle(
        "Position-within-context: does proximity to a gene boundary hurt performance?",
        fontsize=11,
    )
    fig.tight_layout()
    out_path = out_dir / "position_in_context.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[exp2a]   wrote {out_path.name}", file=sys.stderr)

    return summary


# ---------------------------------------------------------------------------
# Experiment 3: error overlap
# ---------------------------------------------------------------------------

def experiment_overlap(data: dict, out_dir: Path, thresh: float = 0.5) -> dict:
    """Compute set overlap of true sites each model gets wrong."""
    print(f"[exp3] error overlap @ thresh={thresh} …", file=sys.stderr)
    sm = data["sm_probs_rescaled"]
    sa = data["sa_probs"]
    labels = data["labels"]

    try:
        from matplotlib_venn import venn2  # noqa: F401
        have_venn = True
    except ImportError:
        have_venn = False
        print("[exp3]   matplotlib_venn unavailable, falling back to bar chart",
              file=sys.stderr)

    summary = {}
    for cls_idx, cls_name in CLASS_NAMES.items():
        mask = labels == cls_idx
        n = int(mask.sum())
        if n == 0:
            continue

        # Per-class "correct prediction" means assigning high probability to the
        # true class. A "miss" is below the threshold.
        sm_score = sm[mask, cls_idx]
        sa_score = sa[mask, cls_idx]
        miss_sa = sa_score < thresh
        miss_sm = sm_score < thresh
        both = miss_sa & miss_sm
        only_sa = miss_sa & ~miss_sm
        only_sm = ~miss_sa & miss_sm
        union = miss_sa | miss_sm
        jacc = float(both.sum() / max(union.sum(), 1))
        counts = {
            "n_total": n,
            "miss_spliceai": int(miss_sa.sum()),
            "miss_splicemamba": int(miss_sm.sum()),
            "miss_both": int(both.sum()),
            "miss_only_spliceai": int(only_sa.sum()),
            "miss_only_splicemamba": int(only_sm.sum()),
            "jaccard": jacc,
            "threshold": thresh,
        }
        summary[cls_name] = counts

        fig, ax = plt.subplots(figsize=(6.2, 5.2))
        if have_venn:
            from matplotlib_venn import venn2
            venn2(
                subsets=(int(only_sa.sum()), int(only_sm.sum()), int(both.sum())),
                set_labels=("SpliceAI misses", "SpliceMamba misses"),
                ax=ax,
            )
        else:
            bars = ["both miss", "only SpliceAI", "only SpliceMamba"]
            vals = [int(both.sum()), int(only_sa.sum()), int(only_sm.sum())]
            ax.barh(bars, vals, color=["tab:red", "tab:orange", "tab:blue"])
            for i, v in enumerate(vals):
                ax.text(v, i, f" {v:,}", va="center", fontsize=9)
            ax.invert_yaxis()
            ax.set_xlabel(f"# true {cls_name} positions missed (P<{thresh})")

        ax.set_title(
            f"True {cls_name}  —  n={n:,},  Jaccard of misses = {jacc:.3f}"
        )
        fig.tight_layout()
        out_path = out_dir / f"error_overlap_{cls_name}.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"[exp3]   wrote {out_path.name}", file=sys.stderr)

    return summary


# ---------------------------------------------------------------------------
# Experiment 4: calibration / reliability diagrams
# ---------------------------------------------------------------------------

def _reliability(scores: np.ndarray, binary_true: np.ndarray, n_bins: int = 10):
    """Return (mean_pred, empirical_rate, counts) for each non-empty bin."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_idx = np.clip(np.digitize(scores, edges) - 1, 0, n_bins - 1)
    mean_pred = np.zeros(n_bins)
    emp = np.zeros(n_bins)
    counts = np.zeros(n_bins, dtype=np.int64)
    for b in range(n_bins):
        m = bin_idx == b
        c = int(m.sum())
        counts[b] = c
        if c > 0:
            mean_pred[b] = float(scores[m].mean())
            emp[b] = float(binary_true[m].mean())
    return mean_pred, emp, counts, edges


def _ece_restricted(scores, true, lo, hi, n_bins):
    """ECE restricted to bins whose predicted mean is in [lo, hi)."""
    mean_pred, emp, counts, _ = _reliability(scores, true, n_bins=n_bins)
    keep = (mean_pred >= lo) & (mean_pred < hi) & (counts > 0)
    if not keep.any():
        return float("nan"), int(0)
    kept_counts = counts[keep]
    total = kept_counts.sum()
    ece = float(np.sum(kept_counts / total * np.abs(mean_pred[keep] - emp[keep])))
    return ece, int(total)


def experiment_calibration(data: dict, out_dir: Path, n_bins: int = 10) -> dict:
    """Reliability diagrams + ECE + Brier per class, per model.

    We produce TWO rows of plots per class:
      * Top: raw probabilities (what the model actually outputs).
      * Bottom: SpliceMamba rescaled to [0, 1] per-class (for fair comparison
        when users interpret threshold values identically across models).

    Also report ECE_actionable = ECE restricted to bins where the mean predicted
    probability exceeds 0.3 (i.e., where the model is making a meaningful
    positive call). SpliceAI's 'overconfidence' will show up there if present.
    """
    print("[exp4] calibration / reliability …", file=sys.stderr)
    sm_raw = data["sm_probs"]
    sm_resc = data["sm_probs_rescaled"]
    sa = data["sa_probs"]
    labels = data["labels"]

    summary = {}
    fig, axes = plt.subplots(2, 2, figsize=(13, 11))

    for row, variant in enumerate(["raw", "rescaled"]):
        sm = sm_raw if variant == "raw" else sm_resc
        for col, (cls_idx, cls_name) in enumerate([(1, "acceptor"), (2, "donor")]):
            ax = axes[row, col]
            true = (labels == cls_idx).astype(np.int32)

            cls_summary = summary.setdefault(cls_name, {}).setdefault(variant, {})
            for key, probs, colour, lbl in [
                ("sa", sa, "tab:orange", "SpliceAI"),
                ("sm", sm, "tab:blue",
                    "SpliceMamba" + (" (raw)" if variant == "raw" else " (rescaled)")),
            ]:
                scores = probs[:, cls_idx].astype(np.float64)
                mean_pred, emp, counts, _ = _reliability(scores, true, n_bins=n_bins)
                total = counts.sum()
                ece = float(
                    np.sum(counts / max(total, 1) * np.abs(mean_pred - emp))
                )
                brier = float(brier_score_loss(true, scores))
                # ECE restricted to "actionable" predictions (pred > 0.3)
                ece_act, n_act = _ece_restricted(scores, true, 0.3, 1.01, n_bins)
                non_empty = counts > 0
                ax.plot(
                    mean_pred[non_empty], emp[non_empty], "o-", color=colour,
                    label=(
                        f"{lbl}\n"
                        f"  ECE(all) = {ece:.4f}   Brier = {brier:.6f}\n"
                        f"  ECE(pred≥0.3, n={n_act:,}) = "
                        + (f"{ece_act:.4f}" if not np.isnan(ece_act) else "n/a")
                    ),
                )
                # Annotate counts above each marker (log scale, tiny)
                for mp, em, cn in zip(mean_pred[non_empty], emp[non_empty],
                                      counts[non_empty]):
                    ax.annotate(f"{cn:,}", (mp, em), fontsize=6,
                                textcoords="offset points", xytext=(3, 3),
                                color=colour, alpha=0.7)
                cls_summary[key] = {
                    "ECE_all": ece,
                    "Brier": brier,
                    "ECE_actionable_pred_ge_0.3": ece_act,
                    "n_actionable_positions": n_act,
                    "mean_pred": mean_pred.tolist(),
                    "empirical": emp.tolist(),
                    "counts": counts.tolist(),
                }

            ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.6)
            ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
            ax.set_xlabel(f"Mean predicted P({cls_name})")
            ax.set_ylabel("Empirical positive frequency")
            ax.set_title(f"{cls_name.capitalize()} — {variant} probabilities")
            ax.legend(loc="upper left", fontsize=7, framealpha=0.9)
            ax.grid(True, alpha=0.3)

    fig.suptitle(
        "Calibration: predicted probability vs empirical positive rate  "
        "(closer to y=x is better; overconfidence → curve below y=x)",
        fontsize=11,
    )
    fig.tight_layout()
    out_path = out_dir / "calibration.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[exp4]   wrote {out_path.name}", file=sys.stderr)

    return summary


# ---------------------------------------------------------------------------
# Main dispatch
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment",
                    choices=["scatter", "position", "overlap",
                             "calibration", "all"],
                    default="all")
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="Decision threshold for error-overlap (exp3)")
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = load_predictions()

    summary = {"sm_rescale": data["sm_rescale"]}

    if args.experiment in ("scatter", "all"):
        summary["experiment_1_scatter"] = experiment_scatter(data, out_dir)
    if args.experiment in ("position", "all"):
        summary["experiment_2a_position"] = experiment_position(data, out_dir)
    if args.experiment in ("overlap", "all"):
        summary["experiment_3_overlap"] = experiment_overlap(
            data, out_dir, thresh=args.threshold)
    if args.experiment in ("calibration", "all"):
        summary["experiment_4_calibration"] = experiment_calibration(data, out_dir)

    summary_path = out_dir / "behavior_summary.json"
    with open(summary_path, "w") as f:
        json.dump(make_serializable(summary), f, indent=2)
    print(f"[done] wrote summary → {summary_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
