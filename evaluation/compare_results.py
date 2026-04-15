"""
Compare SpliceMamba, SpliceAI, and Pangolin evaluation results.

Loads JSON result files from all models and produces:
  - 3-way binary splice metrics table (all models, fair comparison)
  - 2-way 3-class metrics table (SpliceMamba vs SpliceAI, donor/acceptor)
  - Tissue-specific comparison (if tissue results available)
  - Plots: bar chart, threshold sweep, stratified performance
  - CSV export

Usage:
    python compare_results.py
    python compare_results.py \\
        --splicemamba results/splicemamba_results.json \\
        --spliceai results/spliceai_results.json \\
        --pangolin results/pangolin_results.json \\
        --tissue results/tissue_specific_results.json \\
        --output-dir results/comparison/
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


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

    print(f"\nAll outputs saved to {output_dir}/")


if __name__ == "__main__":
    main()
