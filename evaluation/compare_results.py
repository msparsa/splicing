"""
Compare SpliceMamba, SpliceAI, and Pangolin evaluation results.

Loads JSON result files from all models and produces:
  - 3-way binary splice metrics table (all models, fair comparison)
  - 2-way 3-class metrics table (SpliceMamba vs SpliceAI, donor/acceptor)
  - Tissue-specific comparison (if tissue results available)
  - Plots: bar chart, threshold sweep, stratified performance
  - CSV export

Usage:
    python evaluation/compare_results.py
    python evaluation/compare_results.py \\
        --splicemamba evaluation/results/splicemamba_results.json \\
        --spliceai evaluation/results/spliceai_results.json \\
        --pangolin evaluation/results/pangolin_results.json \\
        --tissue evaluation/results/tissue_specific_results.json \\
        --output-dir evaluation/results/comparison/
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np

import sys
_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from eval_utils import (
    compute_gene_window_counts,
    stitch_gene_predictions,
    read_window_labels,
    stitch_gene_labels,
    adapt_to_binary_splice,
    labels_to_binary,
    parse_gene_junctions,
    compute_stratified_auprc_topk,
    INTRON_BUCKETS,
    EXON_BUCKETS,
)


# ---------------------------------------------------------------------------
# Load results
# ---------------------------------------------------------------------------

def load_results(path: str) -> dict | None:
    p = Path(path)
    if not p.exists():
        print(f"  WARNING: {path} not found, skipping")
        return None
    with open(p) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Extract binary metrics (unified format from any model)
# ---------------------------------------------------------------------------

def extract_binary_metrics(results: dict) -> dict | None:
    """Normalize binary metrics from any model's JSON format."""
    if results is None:
        return None

    model = results.get("model", "")

    if model == "pangolin":
        avg = results.get("results_per_tissue", {}).get("averaged")
        if avg is None:
            # Fall back to first tissue
            tissues = results.get("results_per_tissue", {})
            for k, v in tissues.items():
                if isinstance(v, dict) and "auprc" in v:
                    avg = v
                    break
        if avg is None:
            return None
        return {
            "auprc": avg["auprc"],
            "topk": avg["topk"],
            "f1": avg["f1"],
            "positional": avg.get("positional", {}),
        }
    else:
        # SpliceMamba or SpliceAI — binary metrics under metrics.binary
        binary = results.get("metrics", {}).get("binary")
        if binary is None:
            return None
        return binary


# ---------------------------------------------------------------------------
# 3-class comparison (SpliceMamba vs SpliceAI)
# ---------------------------------------------------------------------------

def build_3class_rows(
    mamba: dict, spliceai: dict,
) -> list[tuple[str, float, float, float]]:
    """Return (metric_name, mamba_val, spliceai_val, delta) tuples."""
    rows = []
    m = mamba["metrics"]
    s = spliceai["metrics"]

    for key in ["auprc_donor", "auprc_acceptor", "auprc_mean"]:
        rows.append((key, m["auprc"][key], s["auprc"][key],
                      m["auprc"][key] - s["auprc"][key]))

    for key in ["topk_global_donor", "topk_global_acceptor", "topk_global_mean"]:
        rows.append((key, m["topk"][key], s["topk"][key],
                      m["topk"][key] - s["topk"][key]))

    for cls in ["donor", "acceptor"]:
        key = f"f1_{cls}_best"
        rows.append((key, m["f1_optimal"][key], s["f1_optimal"][key],
                      m["f1_optimal"][key] - s["f1_optimal"][key]))

    for cls in ["donor", "acceptor"]:
        for metric in ["mean_offset", "within_1bp", "within_5bp"]:
            key = f"positional_{cls}_{metric}"
            mv = m["positional"][key]
            sv = s["positional"][key]
            mv = float(mv) if not isinstance(mv, str) else float("inf")
            sv = float(sv) if not isinstance(sv, str) else float("inf")
            delta = mv - sv if not (np.isinf(mv) or np.isinf(sv)) else float("nan")
            rows.append((key, mv, sv, delta))

    return rows


def build_binary_rows(
    models: dict[str, dict | None],
) -> list[tuple[str, dict[str, float]]]:
    """Return (metric_name, {model: value}) for binary metrics."""
    metrics_map = [
        ("AUPRC (splice)", "auprc", "auprc_splice"),
        ("Top-k (global)", "topk", "topk_global_splice"),
        ("Top-k (k=1)", "topk", "topk_splice_k1"),
        ("Top-k (k=0.5)", "topk", "topk_splice_k0.5"),
        ("F1 (best)", "f1", "f1_splice_best"),
        ("Within +/-1bp", "positional", "positional_splice_within_1bp"),
        ("Within +/-5bp", "positional", "positional_splice_within_5bp"),
        ("Mean offset (bp)", "positional", "positional_splice_mean_offset"),
    ]

    rows = []
    for label, section, key in metrics_map:
        vals = {}
        for model_name, binary in models.items():
            if binary is not None and section in binary and key in binary[section]:
                v = binary[section][key]
                vals[model_name] = float(v) if not isinstance(v, str) else float("inf")
            else:
                vals[model_name] = None
        rows.append((label, vals))
    return rows


# ---------------------------------------------------------------------------
# Print tables
# ---------------------------------------------------------------------------

def print_binary_table(
    rows: list[tuple[str, dict[str, float]]],
    model_order: list[str],
):
    """Print 3-way binary metrics table."""
    col_width = 14
    header = f"{'Metric':<25}"
    for m in model_order:
        header += f" {m:>{col_width}}"
    print("\n" + "=" * (25 + (col_width + 1) * len(model_order)))
    print("BINARY SPLICE METRICS (All Models)")
    print("=" * (25 + (col_width + 1) * len(model_order)))
    print(header)
    print("-" * (25 + (col_width + 1) * len(model_order)))
    for label, vals in rows:
        row = f"{label:<25}"
        for m in model_order:
            v = vals.get(m)
            if v is None:
                row += f" {'N/A':>{col_width}}"
            elif np.isinf(v):
                row += f" {'inf':>{col_width}}"
            else:
                row += f" {v:>{col_width}.4f}"
        print(row)
    print("=" * (25 + (col_width + 1) * len(model_order)))


def print_3class_table(rows: list[tuple[str, float, float, float]]):
    """Print 2-way 3-class metrics table."""
    print("\n" + "=" * 72)
    print("3-CLASS METRICS (Donor/Acceptor — SpliceMamba vs SpliceAI)")
    print("=" * 72)
    print(f"{'Metric':<35} {'SpliceMamba':>12} {'SpliceAI':>12} {'Delta':>10}")
    print("-" * 72)
    for name, mv, sv, delta in rows:
        mv_str = f"{mv:.4f}" if not np.isinf(mv) else "inf"
        sv_str = f"{sv:.4f}" if not np.isinf(sv) else "inf"
        d_str = f"{delta:+.4f}" if not np.isnan(delta) else "n/a"
        print(f"{name:<35} {mv_str:>12} {sv_str:>12} {d_str:>10}")
    print("=" * 72)


def print_tissue_table(tissue_results: dict):
    """Print tissue-specific results for all models.

    Handles both old format (per_tissue) and new format (annotation_based).
    """
    models_data = tissue_results.get("models", {})
    if not models_data:
        return

    # Detect format: new (annotation_based) vs old (per_tissue)
    sample_model = next(iter(models_data.values()))
    if "annotation_based" in sample_model:
        # Collect tissue keys from tissue-specific models
        tissue_keys = set()
        for model_data in models_data.values():
            annot = model_data.get("annotation_based", {})
            tissue_keys.update(
                k for k in annot if k not in ("averaged", "all")
            )
        tissue_keys = sorted(tissue_keys)
        ordered_keys = tissue_keys + ["averaged"]

        print("\n" + "=" * 72)
        print("ANNOTATION-BASED EVALUATION (Pangolin Paper Methodology)")
        print("=" * 72)
        header = f"{'Model':<15}"
        for k in ordered_keys:
            header += f" {k:>12}"
        print(header)
        print("-" * (15 + 13 * len(ordered_keys)))

        for model_name, model_data in models_data.items():
            annot = model_data.get("annotation_based", {})
            # Fallback for tissue-agnostic models
            fallback = annot.get("all", {}).get(
                "auprc", {}
            ).get("auprc_splice", None)

            row = f"{model_name:<15}"
            for key in ordered_keys:
                if key in annot:
                    auprc = annot[key].get("auprc", {}).get("auprc_splice")
                else:
                    auprc = fallback
                if auprc is not None:
                    row += f" {auprc:>12.4f}"
                else:
                    row += f" {'N/A':>12}"
            print(row)
        print("=" * (15 + 13 * len(ordered_keys)))
    else:
        # Old format: per_tissue keys
        tissues = tissue_results.get("tissues", [])
        if not tissues:
            return
        print("\n" + "=" * 72)
        print("TISSUE-SPECIFIC BINARY SPLICE AUPRC")
        print("=" * 72)
        header = f"{'Model':<15}"
        for t in tissues:
            header += f" {t:>12}"
        print(header)
        print("-" * (15 + 13 * len(tissues)))
        for model_name, model_data in models_data.items():
            row = f"{model_name:<15}"
            for t in tissues:
                pt = model_data.get("per_tissue", {}).get(t, {})
                auprc = pt.get("auprc", {}).get("auprc_splice")
                if auprc is not None:
                    row += f" {auprc:>12.4f}"
                else:
                    row += f" {'N/A':>12}"
            print(row)
        print("=" * (15 + 13 * len(tissues)))


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_binary_overview(
    rows: list[tuple[str, dict[str, float]]],
    model_order: list[str],
    output_dir: Path,
):
    """Grouped bar chart of binary metrics for all models."""
    # Select metrics that are on a 0-1 scale
    plot_metrics = [r for r in rows if "offset" not in r[0].lower()]

    labels = [r[0] for r in plot_metrics]
    x = np.arange(len(labels))
    n_models = len(model_order)
    width = 0.7 / n_models

    colors = {
        "SpliceMamba": "#4C72B0",
        "SpliceAI": "#DD8452",
        "Pangolin": "#55A868",
    }

    fig, ax = plt.subplots(figsize=(14, 6))
    for i, model in enumerate(model_order):
        vals = []
        for _, v_dict in plot_metrics:
            v = v_dict.get(model)
            vals.append(v if v is not None else 0)
        offset = (i - n_models / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width,
                       label=model, color=colors.get(model, f"C{i}"))
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005,
                        f"{h:.3f}", ha="center", va="bottom", fontsize=7)

    ax.set_ylabel("Score")
    ax.set_title("Binary Splice Metrics: 3-Way Comparison", fontsize=14,
                  fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9, rotation=15, ha="right")
    ax.set_ylim(0, 1.12)
    ax.legend(fontsize=11)
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    path = output_dir / "metrics_overview_3way.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def plot_tissue_comparison(tissue_results: dict, output_dir: Path):
    """Bar chart of annotation-based AUPRC for all models.

    For Pangolin (tissue-specific), shows per-tissue model + averaged.
    For tissue-agnostic models, shows single bar.
    """
    models_data = tissue_results.get("models", {})
    if not models_data:
        return

    colors = {
        "splicemamba": "#4C72B0",
        "spliceai": "#DD8452",
        "pangolin": "#55A868",
    }

    # Detect format
    sample_model = next(iter(models_data.values()))
    if "annotation_based" in sample_model:
        # Collect tissue keys from Pangolin (tissue-specific model)
        tissue_keys = set()
        for model_data in models_data.values():
            annot = model_data.get("annotation_based", {})
            tissue_keys.update(
                k for k in annot if k not in ("averaged", "all")
            )
        tissue_keys = sorted(tissue_keys)

        # X-axis: individual tissues + averaged
        ordered_keys = tissue_keys + ["averaged"]

        x = np.arange(len(ordered_keys))
        model_names = list(models_data.keys())
        n_models = len(model_names)
        width = 0.7 / n_models

        fig, ax = plt.subplots(figsize=(10, 6))
        for i, model_name in enumerate(model_names):
            vals = []
            annot = models_data[model_name].get("annotation_based", {})

            # For tissue-agnostic models (only "all" key), repeat
            # their single value across all tissue columns
            fallback = annot.get("all", {}).get(
                "auprc", {}
            ).get("auprc_splice", 0)

            for key in ordered_keys:
                if key in annot:
                    auprc = annot[key].get("auprc", {}).get(
                        "auprc_splice", 0
                    )
                else:
                    # Tissue-agnostic: same predictions for every tissue
                    auprc = fallback
                vals.append(auprc)

            offset = (i - n_models / 2 + 0.5) * width
            bars = ax.bar(x + offset, vals, width,
                           label=model_name, color=colors.get(model_name, f"C{i}"))
            for bar in bars:
                h = bar.get_height()
                if h > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005,
                            f"{h:.3f}", ha="center", va="bottom", fontsize=8)

        ax.set_ylabel("Binary Splice AUPRC")
        ax.set_title("Annotation-Based Splice AUPRC (Pangolin Paper Method)",
                      fontsize=14, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([k.capitalize() for k in ordered_keys], fontsize=11)
    else:
        # Old format: per_tissue
        tissues = tissue_results.get("tissues", [])
        if not tissues:
            return
        x = np.arange(len(tissues))
        model_names = list(models_data.keys())
        n_models = len(model_names)
        width = 0.7 / n_models

        fig, ax = plt.subplots(figsize=(10, 6))
        for i, model_name in enumerate(model_names):
            vals = []
            for t in tissues:
                pt = models_data[model_name].get("per_tissue", {}).get(t, {})
                auprc = pt.get("auprc", {}).get("auprc_splice", 0)
                vals.append(auprc)
            offset = (i - n_models / 2 + 0.5) * width
            bars = ax.bar(x + offset, vals, width,
                           label=model_name, color=colors.get(model_name, f"C{i}"))
            for bar in bars:
                h = bar.get_height()
                if h > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005,
                            f"{h:.3f}", ha="center", va="bottom", fontsize=8)

        ax.set_ylabel("Binary Splice AUPRC")
        ax.set_title("Tissue-Specific Splice AUPRC", fontsize=14,
                      fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([t.capitalize() for t in tissues], fontsize=11)

    ax.set_ylim(0, 1.12)
    ax.legend(fontsize=11)
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    path = output_dir / "tissue_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def plot_threshold_sweep(mamba: dict, spliceai: dict, output_dir: Path):
    """Threshold sweep for 3-class models (SpliceMamba vs SpliceAI)."""
    m_sweep = mamba.get("metrics", {}).get("threshold_sweep")
    s_sweep = spliceai.get("metrics", {}).get("threshold_sweep")
    if not m_sweep or not s_sweep:
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Threshold Sweep: SpliceMamba vs SpliceAI", fontsize=14,
                  fontweight="bold")

    for row, cls in enumerate(["donor", "acceptor"]):
        m_data = m_sweep[cls]
        s_data = s_sweep[cls]
        m_thresh = [d["threshold"] for d in m_data]
        s_thresh = [d["threshold"] for d in s_data]

        ax = axes[row, 0]
        ax.plot(m_thresh, [d["precision"] for d in m_data], "b-o",
                markersize=3, label="SpliceMamba Precision")
        ax.plot(m_thresh, [d["recall"] for d in m_data], "b--s",
                markersize=3, label="SpliceMamba Recall")
        ax.plot(s_thresh, [d["precision"] for d in s_data], "r-o",
                markersize=3, label="SpliceAI Precision")
        ax.plot(s_thresh, [d["recall"] for d in s_data], "r--s",
                markersize=3, label="SpliceAI Recall")
        ax.set_xlabel("Threshold")
        ax.set_ylabel("Score")
        ax.set_title(f"{cls.capitalize()} - Precision & Recall")
        ax.legend(fontsize=8)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)

        ax = axes[row, 1]
        m_f1 = [d["f1"] for d in m_data]
        s_f1 = [d["f1"] for d in s_data]
        ax.plot(m_thresh, m_f1, "b-o", markersize=3, label="SpliceMamba")
        ax.plot(s_thresh, s_f1, "r-o", markersize=3, label="SpliceAI")
        m_best_idx = np.argmax(m_f1)
        s_best_idx = np.argmax(s_f1)
        ax.axvline(m_thresh[m_best_idx], color="b", linestyle=":", alpha=0.5)
        ax.axvline(s_thresh[s_best_idx], color="r", linestyle=":", alpha=0.5)
        ax.scatter([m_thresh[m_best_idx]], [m_f1[m_best_idx]], color="b",
                   zorder=5, s=80, marker="*",
                   label=f"Mamba best={m_f1[m_best_idx]:.3f}@{m_thresh[m_best_idx]:.2f}")
        ax.scatter([s_thresh[s_best_idx]], [s_f1[s_best_idx]], color="r",
                   zorder=5, s=80, marker="*",
                   label=f"AI best={s_f1[s_best_idx]:.3f}@{s_thresh[s_best_idx]:.2f}")
        ax.set_xlabel("Threshold")
        ax.set_ylabel("F1 Score")
        ax.set_title(f"{cls.capitalize()} - F1")
        ax.legend(fontsize=8)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = output_dir / "threshold_sweep.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def plot_stratified(mamba: dict, spliceai: dict, output_dir: Path,
                    strat_key: str, title: str, filename: str,
                    bucket_order: list[str]):
    """Stratified performance plot (SpliceMamba vs SpliceAI)."""
    m_strat = mamba.get("metrics", {}).get(strat_key)
    s_strat = spliceai.get("metrics", {}).get(strat_key)
    if not m_strat or not s_strat:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f"Recall@0.5 Stratified by {title}", fontsize=14,
                  fontweight="bold")

    x = np.arange(len(bucket_order))
    width = 0.18

    for col, cls in enumerate(["donor", "acceptor"]):
        ax = axes[col]
        m_vals, s_vals, counts = [], [], []
        for bname in bucket_order:
            m_bucket = m_strat.get(bname, {})
            s_bucket = s_strat.get(bname, {})
            m_vals.append(m_bucket.get(f"{cls}_recall_at_0.5", 0))
            s_vals.append(s_bucket.get(f"{cls}_recall_at_0.5", 0))
            counts.append(m_bucket.get(f"{cls}_n_sites", 0))

        bars1 = ax.bar(x - width / 2, m_vals, width,
                        label="SpliceMamba", color="#4C72B0")
        bars2 = ax.bar(x + width / 2, s_vals, width,
                        label="SpliceAI", color="#DD8452")

        for i, (b1, b2, n) in enumerate(zip(bars1, bars2, counts)):
            y_max = max(b1.get_height(), b2.get_height())
            ax.text(i, y_max + 0.02, f"n={n}", ha="center", fontsize=8,
                    color="gray")

        ax.set_xlabel(title)
        ax.set_ylabel("Recall @ 0.5")
        ax.set_title(f"{cls.capitalize()}")
        ax.set_xticks(x)
        ax.set_xticklabels(bucket_order, rotation=15, ha="right")
        ax.set_ylim(0, 1.15)
        ax.legend()
        ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    path = output_dir / filename
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


# ---------------------------------------------------------------------------
# Prediction-based 3-way comparison plots
# ---------------------------------------------------------------------------

MODEL_COLORS = {
    "SpliceMamba": "#4C72B0",
    "SpliceAI": "#DD8452",
    "Pangolin": "#55A868",
}


def _load_preds_to_gene_binary(
    npz_path: str,
    windows_per_gene: np.ndarray,
    is_3class: bool,
    pangolin_tissue: str | None = None,
) -> list[np.ndarray] | None:
    """Load .npz predictions and return gene-level binary splice probs."""
    p = Path(npz_path)
    if not p.exists():
        print(f"  WARNING: {npz_path} not found, skipping")
        return None
    data = np.load(p)
    if pangolin_tissue:
        if pangolin_tissue not in data:
            # try averaged across tissues
            tissues = [k for k in data.files]
            if not tissues:
                return None
            arr = np.mean([data[t] for t in tissues], axis=0)
        else:
            arr = data[pangolin_tissue]
        # Pangolin: (total_windows, 5000) already binary
        gene_probs = stitch_gene_predictions(arr, windows_per_gene)
        return gene_probs
    else:
        arr = data["probs"]  # (total_windows, 5000, 3)
        gene_probs_3class = stitch_gene_predictions(arr, windows_per_gene)
        if is_3class:
            return adapt_to_binary_splice(gene_probs_3class)
        return gene_probs_3class


def _load_pangolin_averaged(
    npz_path: str,
    windows_per_gene: np.ndarray,
) -> list[np.ndarray] | None:
    """Load Pangolin predictions averaged across all tissues."""
    p = Path(npz_path)
    if not p.exists():
        print(f"  WARNING: {npz_path} not found, skipping")
        return None
    data = np.load(p)
    tissues = [k for k in data.files]
    if not tissues:
        return None
    arr = np.mean([data[t] for t in tissues], axis=0)  # (total_windows, 5000)
    return stitch_gene_predictions(arr, windows_per_gene)


def plot_freq_vs_metric(
    model_stratified: dict[str, dict],
    strat_key: str,
    metric_key: str,
    ylabel: str,
    title: str,
    filename: str,
    bucket_order: list[str],
    output_dir: Path,
):
    """Grouped bar chart: x=bins (with frequency labels), y=metric, 3 models."""
    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(bucket_order))
    model_names = list(model_stratified.keys())
    n_models = len(model_names)
    width = 0.7 / max(n_models, 1)

    for i, model_name in enumerate(model_names):
        strat = model_stratified[model_name].get(strat_key, {})
        vals = []
        counts = []
        for bname in bucket_order:
            bucket = strat.get(bname, {})
            vals.append(bucket.get(metric_key, 0.0))
            counts.append(bucket.get("n_sites", 0))

        offset = (i - n_models / 2 + 0.5) * width
        bars = ax.bar(
            x + offset, vals, width,
            label=model_name,
            color=MODEL_COLORS.get(model_name, f"C{i}"),
        )
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2, h + 0.005,
                    f"{h:.3f}", ha="center", va="bottom", fontsize=7,
                )

    # Add frequency (site count) annotations above all bars
    # Use counts from the first model (same data, same labels)
    first_model = model_names[0]
    first_strat = model_stratified[first_model].get(strat_key, {})
    for i, bname in enumerate(bucket_order):
        n = first_strat.get(bname, {}).get("n_sites", 0)
        ax.text(i, ax.get_ylim()[1] * 0.98, f"n={n}",
                ha="center", va="top", fontsize=9, color="gray",
                fontstyle="italic")

    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(bucket_order, fontsize=10, rotation=15, ha="right")
    ax.set_ylim(0, 1.12)
    ax.legend(fontsize=11)
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    path = output_dir / filename
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def plot_performance_vs_seqlen(
    model_gene_probs: dict[str, list[np.ndarray]],
    gene_labels_binary: list[np.ndarray],
    genes: list[dict],
    output_dir: Path,
):
    """Performance vs gene/sequence length with marginal histograms.

    Main panel: per-gene-length-bin AUPRC for each model.
    Top marginal: histogram of gene lengths.
    Right marginal: histogram of AUPRC values per model.
    """
    from sklearn.metrics import average_precision_score as ap_score

    # Compute gene lengths
    gene_lengths = np.array([g["tx_end"] - g["tx_start"] for g in genes])

    # Create length bins (log-spaced)
    bin_edges = [0, 2000, 5000, 10000, 25000, 50000, 100000, float("inf")]
    bin_labels = ["<2kb", "2-5kb", "5-10kb", "10-25kb", "25-50kb", "50-100kb", ">100kb"]

    # Assign genes to bins
    gene_bin_idx = np.digitize(gene_lengths, bin_edges[1:])  # 0-based bin index

    # Compute per-bin AUPRC for each model
    model_bin_auprc = {}
    for model_name, gene_probs in model_gene_probs.items():
        bin_auprcs = []
        for bi in range(len(bin_labels)):
            gidx = np.where(gene_bin_idx == bi)[0]
            if len(gidx) == 0:
                bin_auprcs.append(np.nan)
                continue
            probs_cat = np.concatenate([gene_probs[g] for g in gidx])
            labels_cat = np.concatenate([gene_labels_binary[g] for g in gidx])
            n_pos = int(labels_cat.sum())
            if n_pos > 0 and n_pos < len(labels_cat):
                bin_auprcs.append(float(ap_score(labels_cat, probs_cat)))
            else:
                bin_auprcs.append(np.nan)
        model_bin_auprc[model_name] = np.array(bin_auprcs)

    # Count genes per bin for marginal histogram
    bin_counts = np.array([int((gene_bin_idx == bi).sum()) for bi in range(len(bin_labels))])

    # --- Figure with marginal histograms ---
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(
        2, 2, width_ratios=[4, 1], height_ratios=[1, 4],
        hspace=0.05, wspace=0.05,
    )

    ax_main = fig.add_subplot(gs[1, 0])
    ax_hist_top = fig.add_subplot(gs[0, 0], sharex=ax_main)
    ax_hist_right = fig.add_subplot(gs[1, 1], sharey=ax_main)

    x = np.arange(len(bin_labels))

    # Main panel: line plot of AUPRC per bin for each model
    for model_name, bin_auprcs in model_bin_auprc.items():
        color = MODEL_COLORS.get(model_name, "gray")
        mask = ~np.isnan(bin_auprcs)
        ax_main.plot(
            x[mask], bin_auprcs[mask], "-o", color=color, label=model_name,
            linewidth=2, markersize=8,
        )

    ax_main.set_xlabel("Gene Length", fontsize=12)
    ax_main.set_ylabel("Binary Splice AUPRC", fontsize=12)
    ax_main.set_xticks(x)
    ax_main.set_xticklabels(bin_labels, fontsize=10, rotation=15, ha="right")
    ax_main.set_ylim(0, 1.05)
    ax_main.legend(fontsize=11)
    ax_main.grid(True, alpha=0.3)

    # Top marginal: gene count histogram
    ax_hist_top.bar(x, bin_counts, color="lightgray", edgecolor="gray", width=0.7)
    for i, c in enumerate(bin_counts):
        ax_hist_top.text(i, c + max(bin_counts) * 0.02, str(c),
                         ha="center", fontsize=8, color="gray")
    ax_hist_top.set_ylabel("# Genes", fontsize=10)
    ax_hist_top.set_title(
        "Model Performance vs Gene Length (with frequency distribution)",
        fontsize=14, fontweight="bold",
    )
    plt.setp(ax_hist_top.get_xticklabels(), visible=False)
    ax_hist_top.grid(True, axis="y", alpha=0.3)

    # Right marginal: AUPRC distribution per model
    for model_name, bin_auprcs in model_bin_auprc.items():
        color = MODEL_COLORS.get(model_name, "gray")
        valid = bin_auprcs[~np.isnan(bin_auprcs)]
        if len(valid) > 1:
            ax_hist_right.hist(
                valid, bins=10, orientation="horizontal",
                color=color, alpha=0.4, label=model_name, edgecolor=color,
            )
    ax_hist_right.set_xlabel("Count", fontsize=10)
    plt.setp(ax_hist_right.get_yticklabels(), visible=False)
    ax_hist_right.grid(True, axis="x", alpha=0.3)

    # Hide upper-right corner
    ax_corner = fig.add_subplot(gs[0, 1])
    ax_corner.axis("off")

    plt.savefig(
        output_dir / "performance_vs_seqlen.png",
        dpi=150, bbox_inches="tight",
    )
    plt.close(fig)
    print(f"Saved {output_dir / 'performance_vs_seqlen.png'}")


def plot_performance_vs_threshold(
    model_gene_probs: dict[str, list[np.ndarray]],
    gene_labels_binary: list[np.ndarray],
    output_dir: Path,
):
    """3-way precision/recall/F1 vs threshold for binary splice detection.

    All models use binary P(splice) vs binary labels.
    """
    from sklearn.metrics import precision_score, recall_score, f1_score

    all_labels = np.concatenate(gene_labels_binary, axis=0)

    thresholds = np.arange(0.05, 1.0, 0.025)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Performance vs Threshold (Binary Splice Detection, 3-way)",
                 fontsize=14, fontweight="bold")

    metric_fns = [
        ("Precision", lambda y, p: precision_score(y, p, zero_division=0)),
        ("Recall", lambda y, p: recall_score(y, p, zero_division=0)),
        ("F1 Score", lambda y, p: f1_score(y, p, zero_division=0)),
    ]

    for col, (metric_name, metric_fn) in enumerate(metric_fns):
        ax = axes[col]

        for model_name, gene_probs in model_gene_probs.items():
            color = MODEL_COLORS.get(model_name, "gray")
            all_probs = np.concatenate(gene_probs, axis=0)

            scores = []
            for t in thresholds:
                preds = (all_probs >= t).astype(np.int32)
                scores.append(metric_fn(all_labels, preds))

            ax.plot(thresholds, scores, "-", color=color, label=model_name,
                    linewidth=2)

            # Mark best F1 threshold
            if metric_name == "F1 Score":
                best_idx = np.argmax(scores)
                ax.scatter([thresholds[best_idx]], [scores[best_idx]],
                           color=color, s=80, marker="*", zorder=5)
                ax.annotate(
                    f"{scores[best_idx]:.3f}@{thresholds[best_idx]:.2f}",
                    (thresholds[best_idx], scores[best_idx]),
                    textcoords="offset points", xytext=(5, 10),
                    fontsize=8, color=color,
                )

        # Mark key thresholds
        for t in [0.5, 0.95]:
            ax.axvline(t, color="gray", linestyle=":", alpha=0.4, linewidth=0.8)

        ax.set_xlabel("Threshold", fontsize=12)
        ax.set_ylabel(metric_name, fontsize=12)
        ax.set_title(metric_name, fontsize=13)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = output_dir / "performance_vs_threshold_3way.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

def save_3way_csv(
    binary_rows: list[tuple[str, dict[str, float]]],
    class3_rows: list[tuple[str, float, float, float]] | None,
    model_order: list[str],
    path: Path,
):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)

        # Binary section
        writer.writerow(["section", "metric"] + model_order)
        for label, vals in binary_rows:
            row = ["binary", label]
            for m in model_order:
                v = vals.get(m)
                row.append(f"{v:.6f}" if v is not None and not np.isinf(v) else "")
            writer.writerow(row)

        # 3-class section
        if class3_rows:
            writer.writerow([])
            writer.writerow(["section", "metric", "splicemamba", "spliceai", "delta"])
            for name, mv, sv, delta in class3_rows:
                mv_s = f"{mv:.6f}" if not np.isinf(mv) else ""
                sv_s = f"{sv:.6f}" if not np.isinf(sv) else ""
                d_s = f"{delta:.6f}" if not np.isnan(delta) else ""
                writer.writerow(["3class", name, mv_s, sv_s, d_s])

    print(f"CSV saved to {path}")


# ---------------------------------------------------------------------------
# Pangolin tissue breakdown
# ---------------------------------------------------------------------------

def print_pangolin_tissues(pangolin: dict):
    """Print per-tissue breakdown for Pangolin."""
    rpt = pangolin.get("results_per_tissue", {})
    tissues = [t for t in rpt if t != "averaged"]
    if not tissues:
        return

    print("\n" + "=" * 72)
    print("PANGOLIN PER-TISSUE BREAKDOWN")
    print("=" * 72)
    print(f"{'Tissue':<12} {'AUPRC':>10} {'Top-k':>10} {'F1':>10} {'Within 1bp':>12}")
    print("-" * 56)
    for tissue in tissues + ["averaged"]:
        data = rpt.get(tissue, {})
        auprc = data.get("auprc", {}).get("auprc_splice", None)
        topk = data.get("topk", {}).get("topk_global_splice", None)
        f1 = data.get("f1", {}).get("f1_splice_best", None)
        pos = data.get("positional", {}).get("positional_splice_within_1bp", None)
        row = f"{tissue:<12}"
        row += f" {auprc:>10.4f}" if auprc else f" {'N/A':>10}"
        row += f" {topk:>10.4f}" if topk else f" {'N/A':>10}"
        row += f" {f1:>10.4f}" if f1 else f" {'N/A':>10}"
        row += f" {pos:>12.1%}" if pos else f" {'N/A':>12}"
        print(row)
    print("=" * 56)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Compare SpliceMamba, SpliceAI, and Pangolin results"
    )
    _results = str(Path(__file__).resolve().parent / "results")
    _repo = str(Path(__file__).resolve().parent.parent)
    parser.add_argument("--splicemamba", type=str,
                        default=f"{_results}/splicemamba_results.json")
    parser.add_argument("--spliceai", type=str,
                        default=f"{_results}/spliceai_results.json")
    parser.add_argument("--pangolin", type=str,
                        default=f"{_results}/pangolin_results.json")
    parser.add_argument("--tissue", type=str,
                        default=f"{_results}/tissue_specific_results.json",
                        help="Tissue-specific results JSON (from evaluate_tissue.py)")
    parser.add_argument("--output-dir", type=str,
                        default=f"{_results}/comparison")
    # Raw prediction files for 3-way stratified comparison
    parser.add_argument("--splicemamba-preds", type=str,
                        default=f"{_results}/splicemamba_ensemble_preds.npz",
                        help="SpliceMamba .npz predictions (from evaluate.py --save-preds)")
    parser.add_argument("--spliceai-preds", type=str,
                        default=f"{_results}/spliceai_preds.npz",
                        help="SpliceAI .npz predictions (from evaluate_spliceai.py --save-preds)")
    parser.add_argument("--pangolin-preds", type=str,
                        default=f"{_results}/pangolin_preds.npz",
                        help="Pangolin .npz predictions (from evaluate_pangolin.py)")
    parser.add_argument("--dataset-path", type=str,
                        default=f"{_repo}/dataset_test_0.h5",
                        help="HDF5 test dataset (for reading labels)")
    parser.add_argument("--datafile-path", type=str,
                        default=f"{_repo}/datafile_test_0.h5",
                        help="HDF5 test datafile (for gene metadata)")
    args = parser.parse_args()

    print("Loading results...")
    mamba = load_results(args.splicemamba)
    spliceai = load_results(args.spliceai)
    pangolin = load_results(args.pangolin)
    tissue = load_results(args.tissue)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Binary metrics (3-way) ---
    binary_models = {}
    model_order = []
    for name, results in [("SpliceMamba", mamba), ("SpliceAI", spliceai),
                           ("Pangolin", pangolin)]:
        bm = extract_binary_metrics(results)
        if bm is not None:
            binary_models[name] = bm
            model_order.append(name)

    if binary_models:
        binary_rows = build_binary_rows(binary_models)
        print_binary_table(binary_rows, model_order)
        plot_binary_overview(binary_rows, model_order, output_dir)
    else:
        binary_rows = []
        print("\nNo binary metrics available for comparison.")

    # --- 3-class metrics (2-way) ---
    class3_rows = None
    if mamba and spliceai:
        has_3class = ("auprc" in mamba.get("metrics", {}) and
                      "auprc" in spliceai.get("metrics", {}))
        if has_3class:
            class3_rows = build_3class_rows(mamba, spliceai)
            print_3class_table(class3_rows)

    # --- Pangolin per-tissue breakdown ---
    if pangolin:
        print_pangolin_tissues(pangolin)

    # --- Tissue-specific comparison ---
    if tissue:
        print_tissue_table(tissue)
        plot_tissue_comparison(tissue, output_dir)

    # --- Threshold sweep & stratified (SpliceMamba vs SpliceAI) ---
    if mamba and spliceai:
        plot_threshold_sweep(mamba, spliceai, output_dir)

        intron_order = ["<200bp", "200-1000bp", "1000-5000bp", ">5000bp"]
        exon_order = ["<80bp", "80-200bp", "200-500bp", ">500bp"]
        plot_stratified(mamba, spliceai, output_dir,
                        "stratified_by_intron_length", "Intron Length",
                        "stratified_intron.png", intron_order)
        plot_stratified(mamba, spliceai, output_dir,
                        "stratified_by_exon_length", "Exon Length",
                        "stratified_exon.png", exon_order)

    # --- CSV export ---
    if binary_rows:
        save_3way_csv(binary_rows, class3_rows, model_order,
                      output_dir / "comparison_3way.csv")

    # ===================================================================
    # Prediction-based 3-way comparison (requires .npz files)
    # ===================================================================
    preds_available = any(
        Path(p).exists() for p in [
            args.splicemamba_preds, args.spliceai_preds, args.pangolin_preds,
        ]
    )
    if preds_available and Path(args.dataset_path).exists():
        print("\n" + "=" * 60)
        print("PREDICTION-BASED 3-WAY COMPARISON")
        print("=" * 60)

        # Load shared data
        windows_per_gene = compute_gene_window_counts(args.datafile_path)
        all_labels = read_window_labels(args.dataset_path)
        gene_labels = stitch_gene_labels(all_labels, windows_per_gene)
        gene_labels_binary = labels_to_binary(gene_labels)
        genes = parse_gene_junctions(args.datafile_path)

        # Load predictions from each model
        model_gene_probs = {}
        if Path(args.splicemamba_preds).exists():
            print("Loading SpliceMamba predictions...")
            gp = _load_preds_to_gene_binary(
                args.splicemamba_preds, windows_per_gene, is_3class=True,
            )
            if gp is not None:
                model_gene_probs["SpliceMamba"] = gp

        if Path(args.spliceai_preds).exists():
            print("Loading SpliceAI predictions...")
            gp = _load_preds_to_gene_binary(
                args.spliceai_preds, windows_per_gene, is_3class=True,
            )
            if gp is not None:
                model_gene_probs["SpliceAI"] = gp

        if Path(args.pangolin_preds).exists():
            print("Loading Pangolin predictions...")
            gp = _load_pangolin_averaged(args.pangolin_preds, windows_per_gene)
            if gp is not None:
                model_gene_probs["Pangolin"] = gp

        if model_gene_probs:
            # Compute stratified AUPRC & top-k for each model
            print("\nComputing stratified metrics from raw predictions...")
            model_stratified = {}
            for model_name, gene_probs in model_gene_probs.items():
                strat = compute_stratified_auprc_topk(
                    gene_probs, gene_labels_binary, genes,
                )
                model_stratified[model_name] = strat
                print(f"  {model_name}: done")

            intron_order = list(INTRON_BUCKETS.keys())
            exon_order = list(EXON_BUCKETS.keys())

            # Figure 1: Frequency vs AUPRC
            plot_freq_vs_metric(
                model_stratified, "by_intron_length", "auprc",
                ylabel="Binary Splice AUPRC",
                title="AUPRC by Intron Length (3-way comparison)",
                filename="freq_vs_auprc_intron.png",
                bucket_order=intron_order,
                output_dir=output_dir,
            )
            plot_freq_vs_metric(
                model_stratified, "by_exon_length", "auprc",
                ylabel="Binary Splice AUPRC",
                title="AUPRC by Exon Length (3-way comparison)",
                filename="freq_vs_auprc_exon.png",
                bucket_order=exon_order,
                output_dir=output_dir,
            )

            # Figure 2: Frequency vs Top-k
            plot_freq_vs_metric(
                model_stratified, "by_intron_length", "topk",
                ylabel="Top-k Accuracy",
                title="Top-k Accuracy by Intron Length (3-way comparison)",
                filename="freq_vs_topk_intron.png",
                bucket_order=intron_order,
                output_dir=output_dir,
            )
            plot_freq_vs_metric(
                model_stratified, "by_exon_length", "topk",
                ylabel="Top-k Accuracy",
                title="Top-k Accuracy by Exon Length (3-way comparison)",
                filename="freq_vs_topk_exon.png",
                bucket_order=exon_order,
                output_dir=output_dir,
            )

            # Figure 3: Performance vs sequence length with marginal histograms
            plot_performance_vs_seqlen(
                model_gene_probs, gene_labels_binary, genes, output_dir,
            )

            # Figure 4: Performance vs threshold (precision/recall/F1)
            plot_performance_vs_threshold(
                model_gene_probs, gene_labels_binary, output_dir,
            )

            print(f"\nPrediction-based plots saved to {output_dir}/")
        else:
            print("No prediction files found, skipping 3-way stratified plots.")
    else:
        if not preds_available:
            print("\nNo .npz prediction files found — skipping stratified 3-way plots.")
            print("  Run evaluate.py --save-preds to generate SpliceMamba predictions.")

    print(f"\nAll outputs saved to {output_dir}/")


if __name__ == "__main__":
    main()
