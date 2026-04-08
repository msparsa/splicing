"""
Compare SpliceMamba and SpliceAI evaluation results.

Loads JSON result files from both models and produces:
  - Side-by-side comparison table (printed + CSV)
  - Threshold sweep curves
  - Stratified performance by intron/exon length
  - Overall metric comparison bar chart

Usage:
    python compare_results.py \\
        --splicemamba results/splicemamba_results.json \\
        --spliceai results/spliceai_results.json \\
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

def load_results(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Comparison table
# ---------------------------------------------------------------------------

def build_comparison_rows(mamba: dict, spliceai: dict) -> list[tuple[str, float, float, float]]:
    """Return list of (metric_name, mamba_val, spliceai_val, delta) tuples."""
    rows = []
    m = mamba["metrics"]
    s = spliceai["metrics"]

    # AUPRC
    for key in ["auprc_donor", "auprc_acceptor", "auprc_mean"]:
        rows.append((key, m["auprc"][key], s["auprc"][key],
                      m["auprc"][key] - s["auprc"][key]))

    # Top-k global
    for key in ["topk_global_donor", "topk_global_acceptor", "topk_global_mean"]:
        rows.append((key, m["topk"][key], s["topk"][key],
                      m["topk"][key] - s["topk"][key]))

    # F1 optimal
    for cls in ["donor", "acceptor"]:
        key = f"f1_{cls}_best"
        rows.append((key, m["f1_optimal"][key], s["f1_optimal"][key],
                      m["f1_optimal"][key] - s["f1_optimal"][key]))

    # Positional accuracy
    for cls in ["donor", "acceptor"]:
        for metric in ["mean_offset", "within_1bp", "within_5bp"]:
            key = f"positional_{cls}_{metric}"
            mv = m["positional"][key]
            sv = s["positional"][key]
            # Handle inf stored as string
            mv = float(mv) if not isinstance(mv, str) else float("inf")
            sv = float(sv) if not isinstance(sv, str) else float("inf")
            delta = mv - sv if not (np.isinf(mv) or np.isinf(sv)) else float("nan")
            rows.append((key, mv, sv, delta))

    return rows


def print_comparison_table(rows: list[tuple[str, float, float, float]]):
    """Print formatted comparison table to console."""
    header = f"{'Metric':<35} {'SpliceMamba':>12} {'SpliceAI':>12} {'Delta':>10}"
    print("\n" + "=" * 72)
    print("MODEL COMPARISON")
    print("=" * 72)
    print(header)
    print("-" * 72)
    for name, mv, sv, delta in rows:
        mv_str = f"{mv:.4f}" if not np.isinf(mv) else "inf"
        sv_str = f"{sv:.4f}" if not np.isinf(sv) else "inf"
        d_str = f"{delta:+.4f}" if not np.isnan(delta) else "n/a"
        print(f"{name:<35} {mv_str:>12} {sv_str:>12} {d_str:>10}")
    print("=" * 72)


def save_comparison_csv(rows: list[tuple[str, float, float, float]], path: Path):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "splicemamba", "spliceai", "delta"])
        for name, mv, sv, delta in rows:
            writer.writerow([name, mv, sv, delta])
    print(f"Comparison CSV saved to {path}")


# ---------------------------------------------------------------------------
# Plot 1: Threshold sweep curves
# ---------------------------------------------------------------------------

def plot_threshold_sweep(mamba: dict, spliceai: dict, output_dir: Path):
    m_sweep = mamba["metrics"]["threshold_sweep"]
    s_sweep = spliceai["metrics"]["threshold_sweep"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Threshold Sweep: SpliceMamba vs SpliceAI", fontsize=14, fontweight="bold")

    for row, cls in enumerate(["donor", "acceptor"]):
        m_data = m_sweep[cls]
        s_data = s_sweep[cls]
        m_thresh = [d["threshold"] for d in m_data]
        s_thresh = [d["threshold"] for d in s_data]

        # Left column: Precision & Recall vs threshold
        ax = axes[row, 0]
        ax.plot(m_thresh, [d["precision"] for d in m_data], "b-o", markersize=3,
                label="SpliceMamba Precision")
        ax.plot(m_thresh, [d["recall"] for d in m_data], "b--s", markersize=3,
                label="SpliceMamba Recall")
        ax.plot(s_thresh, [d["precision"] for d in s_data], "r-o", markersize=3,
                label="SpliceAI Precision")
        ax.plot(s_thresh, [d["recall"] for d in s_data], "r--s", markersize=3,
                label="SpliceAI Recall")
        ax.set_xlabel("Threshold")
        ax.set_ylabel("Score")
        ax.set_title(f"{cls.capitalize()} - Precision & Recall")
        ax.legend(fontsize=8)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)

        # Right column: F1 vs threshold
        ax = axes[row, 1]
        m_f1 = [d["f1"] for d in m_data]
        s_f1 = [d["f1"] for d in s_data]
        ax.plot(m_thresh, m_f1, "b-o", markersize=3, label="SpliceMamba")
        ax.plot(s_thresh, s_f1, "r-o", markersize=3, label="SpliceAI")

        # Mark optimal thresholds
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


# ---------------------------------------------------------------------------
# Plot 2 & 3: Stratified performance by intron/exon length
# ---------------------------------------------------------------------------

def plot_stratified(mamba: dict, spliceai: dict, output_dir: Path,
                    strat_key: str, title: str, filename: str,
                    bucket_order: list[str]):
    m_strat = mamba["metrics"][strat_key]
    s_strat = spliceai["metrics"][strat_key]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f"Recall@0.5 Stratified by {title}", fontsize=14, fontweight="bold")

    x = np.arange(len(bucket_order))
    width = 0.18

    for col, cls in enumerate(["donor", "acceptor"]):
        ax = axes[col]
        m_vals = []
        s_vals = []
        counts = []
        for bname in bucket_order:
            m_bucket = m_strat.get(bname, {})
            s_bucket = s_strat.get(bname, {})
            m_vals.append(m_bucket.get(f"{cls}_recall_at_0.5", 0))
            s_vals.append(s_bucket.get(f"{cls}_recall_at_0.5", 0))
            counts.append(m_bucket.get(f"{cls}_n_sites", 0))

        bars1 = ax.bar(x - width / 2, m_vals, width, label="SpliceMamba", color="#4C72B0")
        bars2 = ax.bar(x + width / 2, s_vals, width, label="SpliceAI", color="#DD8452")

        # Annotate with site counts
        for i, (b1, b2, n) in enumerate(zip(bars1, bars2, counts)):
            y_max = max(b1.get_height(), b2.get_height())
            ax.text(i, y_max + 0.02, f"n={n}", ha="center", fontsize=8, color="gray")

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
# Plot 4: Overall metric comparison
# ---------------------------------------------------------------------------

def plot_metrics_overview(mamba: dict, spliceai: dict, output_dir: Path):
    m = mamba["metrics"]
    s = spliceai["metrics"]

    metric_names = [
        "AUPRC\nDonor", "AUPRC\nAcceptor", "AUPRC\nMean",
        "Top-k\nDonor", "Top-k\nAcceptor",
        "F1\nDonor", "F1\nAcceptor",
    ]
    m_vals = [
        m["auprc"]["auprc_donor"], m["auprc"]["auprc_acceptor"], m["auprc"]["auprc_mean"],
        m["topk"]["topk_global_donor"], m["topk"]["topk_global_acceptor"],
        m["f1_optimal"]["f1_donor_best"], m["f1_optimal"]["f1_acceptor_best"],
    ]
    s_vals = [
        s["auprc"]["auprc_donor"], s["auprc"]["auprc_acceptor"], s["auprc"]["auprc_mean"],
        s["topk"]["topk_global_donor"], s["topk"]["topk_global_acceptor"],
        s["f1_optimal"]["f1_donor_best"], s["f1_optimal"]["f1_acceptor_best"],
    ]

    x = np.arange(len(metric_names))
    width = 0.3

    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width / 2, m_vals, width, label="SpliceMamba", color="#4C72B0")
    bars2 = ax.bar(x + width / 2, s_vals, width, label="SpliceAI", color="#DD8452")

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005,
                    f"{h:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_ylabel("Score")
    ax.set_title("SpliceMamba vs SpliceAI: Key Metrics", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names, fontsize=10)
    ax.set_ylim(0, 1.12)
    ax.legend(fontsize=12)
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    path = output_dir / "metrics_overview.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Compare SpliceMamba vs SpliceAI results")
    parser.add_argument("--splicemamba", type=str, default="results/splicemamba_results.json",
                        help="Path to SpliceMamba results JSON")
    parser.add_argument("--spliceai", type=str, default="results/spliceai_results.json",
                        help="Path to SpliceAI results JSON")
    parser.add_argument("--output-dir", type=str, default="results/comparison",
                        help="Directory for comparison outputs")
    args = parser.parse_args()

    mamba = load_results(args.splicemamba)
    spliceai = load_results(args.spliceai)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Comparison table
    rows = build_comparison_rows(mamba, spliceai)
    print_comparison_table(rows)
    save_comparison_csv(rows, output_dir / "comparison.csv")

    # Plots
    plot_threshold_sweep(mamba, spliceai, output_dir)

    intron_order = ["<200bp", "200-1000bp", "1000-5000bp", ">5000bp"]
    exon_order = ["<80bp", "80-200bp", "200-500bp", ">500bp"]

    plot_stratified(mamba, spliceai, output_dir,
                    "stratified_by_intron_length", "Intron Length",
                    "stratified_intron.png", intron_order)
    plot_stratified(mamba, spliceai, output_dir,
                    "stratified_by_exon_length", "Exon Length",
                    "stratified_exon.png", exon_order)

    plot_metrics_overview(mamba, spliceai, output_dir)

    print(f"\nAll outputs saved to {output_dir}/")


if __name__ == "__main__":
    main()
