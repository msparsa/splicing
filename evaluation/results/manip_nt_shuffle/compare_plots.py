"""Four-way comparison:
  - SpliceAI    on the manipulated nucleotide-shuffle test set
  - SpliceMamba on the manipulated nucleotide-shuffle test set
  - SpliceAI    on the manipulated codon-shuffle test set
  - SpliceMamba on the manipulated codon-shuffle test set
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).parent  # manip_nt_shuffle/
NT = OUT
CODON = OUT.parent / "manip_codon_shuffle"

SAI_NT = json.loads((NT / "spliceai_results.json").read_text())["metrics"]
SMM_NT = json.loads((NT / "splicemamba_results.json").read_text())["metrics"]
SAI_CD = json.loads((CODON / "spliceai_results.json").read_text())["metrics"]
SMM_CD = json.loads((CODON / "splicemamba_results.json").read_text())["metrics"]

SERIES = [
    ("SpliceAI (nt-shuffle)", SAI_NT, "#d97706"),
    ("SpliceMamba (nt-shuffle)", SMM_NT, "#2563eb"),
    ("SpliceAI (codon-shuffle)", SAI_CD, "#9a3412"),
    ("SpliceMamba (codon-shuffle)", SMM_CD, "#16a34a"),
]


def bar_group(ax, labels, values_per_series, ylabel, title, ylim=None, fmt="{:.3f}"):
    """values_per_series: list of (series_label, values, color)."""
    x = np.arange(len(labels))
    n = len(values_per_series)
    w = 0.8 / n
    for i, (lab, vals, color) in enumerate(values_per_series):
        offset = (i - (n - 1) / 2) * w
        bars = ax.bar(x + offset, vals, w, label=lab, color=color)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    fmt.format(v), ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    if ylim:
        ax.set_ylim(*ylim)


def get_vals(metric_path, series_metrics_list):
    """Walk metric_path (dot-style) for each series and return list of values."""
    out = []
    for m in series_metrics_list:
        cur = m
        for key in metric_path.split("."):
            cur = cur[key]
        out.append(cur)
    return out


# ---------------------------------------------------------------------------
# Chart 1: Headline metrics
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

bar_group(
    axes[0], ["donor", "acceptor", "mean"],
    [(lab, [m["auprc"]["auprc_donor"], m["auprc"]["auprc_acceptor"], m["auprc"]["auprc_mean"]], c)
     for lab, m, c in SERIES],
    "AUPRC", "AUPRC", ylim=(0, 0.75),
)

bar_group(
    axes[1], ["donor F1", "acceptor F1"],
    [(lab, [m["f1_optimal"]["f1_donor_best"], m["f1_optimal"]["f1_acceptor_best"]], c)
     for lab, m, c in SERIES],
    "F1 (optimal threshold)", "Best F1", ylim=(0, 0.75),
)

bar_group(
    axes[2], ["donor", "acceptor", "mean"],
    [(lab, [m["topk"]["topk_global_donor"], m["topk"]["topk_global_acceptor"], m["topk"]["topk_global_mean"]], c)
     for lab, m, c in SERIES],
    "Top-k accuracy", "Global top-k accuracy", ylim=(0, 0.75),
)

fig.suptitle("Four-way comparison — manipulated test sets", fontsize=13)
fig.tight_layout()
fig.savefig(OUT / "compare_headline.png", dpi=150)
plt.close(fig)


# ---------------------------------------------------------------------------
# Chart 2: PR curves
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
markers = ["o", "s", "D", "^"]
for ax, klass in zip(axes, ["acceptor", "donor"]):
    for (lab, m, c), marker in zip(SERIES, markers):
        sweep = m["threshold_sweep"][klass]
        ps = [r["precision"] for r in sweep]
        rs = [r["recall"] for r in sweep]
        ax.plot(rs, ps, marker=marker, color=c, label=lab, lw=2, ms=4, linestyle="-")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"PR curve — {klass}")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

fig.suptitle("Precision–Recall curves (threshold sweep)", fontsize=13)
fig.tight_layout()
fig.savefig(OUT / "compare_pr_curves.png", dpi=150)
plt.close(fig)


# ---------------------------------------------------------------------------
# Chart 3: Top-k at multiple k values
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
ks = ["0.5", "1", "2", "4"]
for ax, klass in zip(axes, ["acceptor", "donor"]):
    series_vals = [(lab, [m["topk"][f"topk_{klass}_k{k}"] for k in ks], c)
                   for lab, m, c in SERIES]
    bar_group(ax, [f"k={k}" for k in ks], series_vals,
              "Top-k accuracy", f"Top-k accuracy — {klass}", ylim=(0, 1.0))

fig.suptitle("Top-k accuracy at multiple thresholds", fontsize=13)
fig.tight_layout()
fig.savefig(OUT / "compare_topk.png", dpi=150)
plt.close(fig)


# ---------------------------------------------------------------------------
# Chart 4: Positional accuracy
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(17, 5))

bar_group(
    axes[0], ["donor", "acceptor"],
    [(lab, [m["positional"]["positional_donor_within_1bp"],
            m["positional"]["positional_acceptor_within_1bp"]], c) for lab, m, c in SERIES],
    "Fraction within 1bp", "Positional accuracy ≤1bp", ylim=(0, 0.75),
)

bar_group(
    axes[1], ["donor", "acceptor"],
    [(lab, [m["positional"]["positional_donor_within_5bp"],
            m["positional"]["positional_acceptor_within_5bp"]], c) for lab, m, c in SERIES],
    "Fraction within 5bp", "Positional accuracy ≤5bp", ylim=(0, 0.75),
)

# log-scale median offset (lower is better)
ax = axes[2]
labels = ["donor", "acceptor"]
x = np.arange(len(labels))
n = len(SERIES)
w = 0.8 / n
for i, (lab, m, c) in enumerate(SERIES):
    raw = [m["positional"]["positional_donor_median_offset"],
           m["positional"]["positional_acceptor_median_offset"]]
    offset = (i - (n - 1) / 2) * w
    bars = ax.bar(x + offset, [v + 1 for v in raw], w, label=lab, color=c)
    for bar, r in zip(bars, raw):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{r:.0f}", ha="center", va="bottom", fontsize=7)
ax.set_yscale("log")
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel("Median offset (bp, log scale, lower is better)")
ax.set_title("Median predicted-vs-true offset")
ax.legend(fontsize=8)
ax.grid(axis="y", alpha=0.3, which="both")

fig.suptitle("Positional accuracy", fontsize=13)
fig.tight_layout()
fig.savefig(OUT / "compare_positional.png", dpi=150)
plt.close(fig)


# ---------------------------------------------------------------------------
# Chart 5: Stratified recall by intron / exon length
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(15, 9))
intron_bins = ["<200bp", "200-1000bp", "1000-5000bp", ">5000bp"]
exon_bins = ["<80bp", "80-200bp", "200-500bp", ">500bp"]

for col, klass in enumerate(["acceptor", "donor"]):
    series_vals = [(lab, [m["stratified_by_intron_length"][b][f"{klass}_recall_at_0.5"]
                          for b in intron_bins], c) for lab, m, c in SERIES]
    bar_group(axes[0, col], intron_bins, series_vals,
              "Recall @ 0.5", f"{klass} — by intron length", ylim=(0, 0.85))

    series_vals = [(lab, [m["stratified_by_exon_length"][b][f"{klass}_recall_at_0.5"]
                          for b in exon_bins], c) for lab, m, c in SERIES]
    bar_group(axes[1, col], exon_bins, series_vals,
              "Recall @ 0.5", f"{klass} — by exon length", ylim=(0, 0.85))

fig.suptitle("Stratified recall (threshold = 0.5)", fontsize=13)
fig.tight_layout()
fig.savefig(OUT / "compare_stratified.png", dpi=150)
plt.close(fig)


# ---------------------------------------------------------------------------
# Chart 6: Per-metric values (grouped bars) — easier to scan than deltas
# ---------------------------------------------------------------------------
metric_specs = [
    ("AUPRC donor", "auprc.auprc_donor"),
    ("AUPRC acceptor", "auprc.auprc_acceptor"),
    ("AUPRC mean", "auprc.auprc_mean"),
    ("F1 donor", "f1_optimal.f1_donor_best"),
    ("F1 acceptor", "f1_optimal.f1_acceptor_best"),
    ("Top-k donor", "topk.topk_global_donor"),
    ("Top-k acceptor", "topk.topk_global_acceptor"),
    ("≤1bp donor", "positional.positional_donor_within_1bp"),
    ("≤1bp acceptor", "positional.positional_acceptor_within_1bp"),
    ("AUPRC binary", "binary.auprc.auprc_splice"),
]
metrics_only = [m for _, m, _ in SERIES]
labels = [name for name, _ in metric_specs]
series_vals = []
for (lab, m, c) in SERIES:
    vals = [get_vals(path, [m])[0] for _, path in metric_specs]
    series_vals.append((lab, vals, c))

fig, ax = plt.subplots(figsize=(14, 6))
x = np.arange(len(labels))
n = len(series_vals)
w = 0.8 / n
for i, (lab, vals, c) in enumerate(series_vals):
    offset = (i - (n - 1) / 2) * w
    bars = ax.bar(x + offset, vals, w, label=lab, color=c)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{v:.3f}", ha="center", va="bottom", fontsize=7, rotation=0)
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=20, ha="right")
ax.set_ylabel("Score (higher = better)")
ax.set_title("All key metrics — three-way comparison")
ax.legend()
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / "compare_summary.png", dpi=150)
plt.close(fig)


print("Saved:")
for name in ["compare_headline.png", "compare_pr_curves.png", "compare_topk.png",
             "compare_positional.png", "compare_stratified.png", "compare_summary.png"]:
    print(f"  {OUT / name}")
