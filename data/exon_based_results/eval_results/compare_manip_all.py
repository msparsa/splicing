"""Cross-experiment comparison across all manip experiments.

Series: each (model, experiment) pair. Both SpliceAI and SpliceMamba evaluated
on every manipulated test set."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).parent
OUT = ROOT / "comparison"
OUT.mkdir(exist_ok=True)

EXPERIMENTS = ["manip_nt_shuffle", "manip_codon_shuffle", "manip_remove1", "manip_remove2"]
SHORT = {
    "manip_nt_shuffle": "nt-shuffle",
    "manip_codon_shuffle": "codon-shuffle",
    "manip_remove1": "remove1",
    "manip_remove2": "remove2",
}

# load every (model, experiment) result
DATA = {}  # (model, exp) -> metrics dict
for exp in EXPERIMENTS:
    for model in ("spliceai", "splicemamba"):
        path = ROOT / exp / f"{model}_results.json"
        DATA[(model, exp)] = json.loads(path.read_text())["metrics"]

EXP_COLORS = {  # one color per experiment
    "manip_nt_shuffle": "#2563eb",
    "manip_codon_shuffle": "#16a34a",
    "manip_remove1": "#d97706",
    "manip_remove2": "#9333ea",
}
MODEL_HATCH = {"spliceai": "//", "splicemamba": ""}
MODEL_FACE = {"spliceai": 0.45, "splicemamba": 1.0}  # alpha


def get(model, exp, path):
    cur = DATA[(model, exp)]
    for k in path.split("."):
        cur = cur[k]
    return cur


# =====================================================================
# Chart 1: AUPRC across experiments, grouped by class
# =====================================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

for ax, klass, key in zip(
    axes,
    ["donor", "acceptor", "mean"],
    ["auprc.auprc_donor", "auprc.auprc_acceptor", "auprc.auprc_mean"],
):
    x = np.arange(len(EXPERIMENTS))
    w = 0.38
    sai_v = [get("spliceai", e, key) for e in EXPERIMENTS]
    smm_v = [get("splicemamba", e, key) for e in EXPERIMENTS]
    sai_colors = [EXP_COLORS[e] for e in EXPERIMENTS]
    b1 = ax.bar(x - w / 2, sai_v, w, color=sai_colors, alpha=0.5,
                hatch="//", edgecolor="black", label="SpliceAI")
    b2 = ax.bar(x + w / 2, smm_v, w, color=sai_colors,
                edgecolor="black", label="SpliceMamba")
    for bar, v in list(zip(b1, sai_v)) + list(zip(b2, smm_v)):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{v:.3f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels([SHORT[e] for e in EXPERIMENTS], rotation=20, ha="right")
    ax.set_title(f"AUPRC ({klass})")
    ax.set_ylabel("AUPRC")
    ax.set_ylim(0, 0.75)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)

fig.suptitle("AUPRC across manipulation experiments — colors = experiment, hatched = SpliceAI", fontsize=13)
fig.tight_layout()
fig.savefig(OUT / "manip_all_auprc.png", dpi=150)
plt.close(fig)


# =====================================================================
# Chart 2: F1 (best threshold) across experiments
# =====================================================================
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
for ax, klass, key in zip(
    axes, ["donor", "acceptor"],
    ["f1_optimal.f1_donor_best", "f1_optimal.f1_acceptor_best"]
):
    x = np.arange(len(EXPERIMENTS))
    w = 0.38
    sai_v = [get("spliceai", e, key) for e in EXPERIMENTS]
    smm_v = [get("splicemamba", e, key) for e in EXPERIMENTS]
    colors = [EXP_COLORS[e] for e in EXPERIMENTS]
    b1 = ax.bar(x - w / 2, sai_v, w, color=colors, alpha=0.5,
                hatch="//", edgecolor="black", label="SpliceAI")
    b2 = ax.bar(x + w / 2, smm_v, w, color=colors, edgecolor="black", label="SpliceMamba")
    for bar, v in list(zip(b1, sai_v)) + list(zip(b2, smm_v)):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{v:.3f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels([SHORT[e] for e in EXPERIMENTS], rotation=20, ha="right")
    ax.set_title(f"Best F1 ({klass})")
    ax.set_ylabel("F1")
    ax.set_ylim(0, 0.75)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)

fig.suptitle("Best F1 across manipulation experiments", fontsize=13)
fig.tight_layout()
fig.savefig(OUT / "manip_all_f1.png", dpi=150)
plt.close(fig)


# =====================================================================
# Chart 3: Top-k accuracy across experiments
# =====================================================================
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
for ax, klass, key in zip(
    axes, ["donor", "acceptor"],
    ["topk.topk_global_donor", "topk.topk_global_acceptor"]
):
    x = np.arange(len(EXPERIMENTS))
    w = 0.38
    sai_v = [get("spliceai", e, key) for e in EXPERIMENTS]
    smm_v = [get("splicemamba", e, key) for e in EXPERIMENTS]
    colors = [EXP_COLORS[e] for e in EXPERIMENTS]
    b1 = ax.bar(x - w / 2, sai_v, w, color=colors, alpha=0.5,
                hatch="//", edgecolor="black", label="SpliceAI")
    b2 = ax.bar(x + w / 2, smm_v, w, color=colors, edgecolor="black", label="SpliceMamba")
    for bar, v in list(zip(b1, sai_v)) + list(zip(b2, smm_v)):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{v:.3f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels([SHORT[e] for e in EXPERIMENTS], rotation=20, ha="right")
    ax.set_title(f"Global top-k accuracy ({klass})")
    ax.set_ylabel("Top-k accuracy")
    ax.set_ylim(0, 0.85)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)

fig.suptitle("Top-k accuracy across manipulation experiments", fontsize=13)
fig.tight_layout()
fig.savefig(OUT / "manip_all_topk.png", dpi=150)
plt.close(fig)


# =====================================================================
# Chart 4: Positional accuracy ≤1bp + median offset (log)
# =====================================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 9))

# row 0: within 1bp
for ax, klass, key in zip(
    axes[0], ["donor", "acceptor"],
    ["positional.positional_donor_within_1bp", "positional.positional_acceptor_within_1bp"]
):
    x = np.arange(len(EXPERIMENTS))
    w = 0.38
    sai_v = [get("spliceai", e, key) for e in EXPERIMENTS]
    smm_v = [get("splicemamba", e, key) for e in EXPERIMENTS]
    colors = [EXP_COLORS[e] for e in EXPERIMENTS]
    b1 = ax.bar(x - w / 2, sai_v, w, color=colors, alpha=0.5,
                hatch="//", edgecolor="black", label="SpliceAI")
    b2 = ax.bar(x + w / 2, smm_v, w, color=colors, edgecolor="black", label="SpliceMamba")
    for bar, v in list(zip(b1, sai_v)) + list(zip(b2, smm_v)):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{v:.3f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels([SHORT[e] for e in EXPERIMENTS], rotation=20, ha="right")
    ax.set_title(f"Fraction ≤1bp ({klass})")
    ax.set_ylabel("Fraction within 1bp")
    ax.set_ylim(0, 0.75)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)

# row 1: median offset (log scale, lower = better)
for ax, klass, key in zip(
    axes[1], ["donor", "acceptor"],
    ["positional.positional_donor_median_offset", "positional.positional_acceptor_median_offset"]
):
    x = np.arange(len(EXPERIMENTS))
    w = 0.38
    sai_v = [get("spliceai", e, key) for e in EXPERIMENTS]
    smm_v = [get("splicemamba", e, key) for e in EXPERIMENTS]
    colors = [EXP_COLORS[e] for e in EXPERIMENTS]
    b1 = ax.bar(x - w / 2, [v + 1 for v in sai_v], w, color=colors, alpha=0.5,
                hatch="//", edgecolor="black", label="SpliceAI")
    b2 = ax.bar(x + w / 2, [v + 1 for v in smm_v], w, color=colors,
                edgecolor="black", label="SpliceMamba")
    for bar, raw in list(zip(b1, sai_v)) + list(zip(b2, smm_v)):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{raw:.0f}", ha="center", va="bottom", fontsize=7)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([SHORT[e] for e in EXPERIMENTS], rotation=20, ha="right")
    ax.set_title(f"Median offset bp ({klass}) — lower = better")
    ax.set_ylabel("Median offset (bp, log)")
    ax.grid(axis="y", alpha=0.3, which="both")
    ax.legend(loc="upper right", fontsize=8)

fig.suptitle("Positional accuracy across manipulation experiments", fontsize=13)
fig.tight_layout()
fig.savefig(OUT / "manip_all_positional.png", dpi=150)
plt.close(fig)


# =====================================================================
# Chart 5: PR curves overlaid — one panel per (model, class)
# =====================================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
for row, model in enumerate(["spliceai", "splicemamba"]):
    for col, klass in enumerate(["acceptor", "donor"]):
        ax = axes[row, col]
        for exp in EXPERIMENTS:
            sweep = DATA[(model, exp)]["threshold_sweep"][klass]
            rs = [r["recall"] for r in sweep]
            ps = [r["precision"] for r in sweep]
            ax.plot(rs, ps, "-o", color=EXP_COLORS[exp], label=SHORT[exp], lw=2, ms=3)
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title(f"{model} — {klass}")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

fig.suptitle("PR curves across manipulation experiments", fontsize=13)
fig.tight_layout()
fig.savefig(OUT / "manip_all_pr_curves.png", dpi=150)
plt.close(fig)


# =====================================================================
# Chart 6: Heatmap — every key metric × every (model, experiment)
# =====================================================================
metrics = [
    ("AUPRC donor",     "auprc.auprc_donor"),
    ("AUPRC acceptor",  "auprc.auprc_acceptor"),
    ("AUPRC mean",      "auprc.auprc_mean"),
    ("F1 donor",        "f1_optimal.f1_donor_best"),
    ("F1 acceptor",     "f1_optimal.f1_acceptor_best"),
    ("Top-k donor",     "topk.topk_global_donor"),
    ("Top-k acceptor",  "topk.topk_global_acceptor"),
    ("≤1bp donor",      "positional.positional_donor_within_1bp"),
    ("≤1bp acceptor",   "positional.positional_acceptor_within_1bp"),
    ("AUPRC binary",    "binary.auprc.auprc_splice"),
]
cols = []
col_labels = []
for exp in EXPERIMENTS:
    for model in ("spliceai", "splicemamba"):
        col_labels.append(f"{model}\n{SHORT[exp]}")
        cols.append([get(model, exp, p) for _, p in metrics])
mat = np.array(cols).T  # rows = metric, cols = (model, exp)

fig, ax = plt.subplots(figsize=(11, 7))
im = ax.imshow(mat, cmap="viridis", aspect="auto", vmin=0, vmax=mat.max())
ax.set_xticks(range(len(col_labels)))
ax.set_xticklabels(col_labels, rotation=30, ha="right", fontsize=9)
ax.set_yticks(range(len(metrics)))
ax.set_yticklabels([m[0] for m in metrics])
for i in range(mat.shape[0]):
    for j in range(mat.shape[1]):
        ax.text(j, i, f"{mat[i, j]:.3f}", ha="center", va="center",
                color="white" if mat[i, j] < mat.max() * 0.5 else "black", fontsize=8)
ax.set_title("All metrics × (model, experiment) — higher (yellow) = better")
fig.colorbar(im, ax=ax, label="score")
fig.tight_layout()
fig.savefig(OUT / "manip_all_heatmap.png", dpi=150)
plt.close(fig)


# =====================================================================
# Chart 7: Δ vs nt-shuffle baseline — does each manip make it harder?
# =====================================================================
baseline = "manip_nt_shuffle"
others = [e for e in EXPERIMENTS if e != baseline]
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for ax, model in zip(axes, ["spliceai", "splicemamba"]):
    rows = []
    for name, key in metrics:
        base = get(model, baseline, key)
        rows.append([get(model, e, key) - base for e in others])
    rows = np.array(rows)
    x = np.arange(len(metrics))
    w = 0.27
    for i, exp in enumerate(others):
        offset = (i - (len(others) - 1) / 2) * w
        bars = ax.bar(x + offset, rows[:, i], w,
                      color=EXP_COLORS[exp], label=SHORT[exp])
        for bar, v in zip(bars, rows[:, i]):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    v + (0.003 if v >= 0 else -0.003),
                    f"{v:+.2f}", ha="center",
                    va="bottom" if v >= 0 else "top", fontsize=6)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([m[0] for m in metrics], rotation=25, ha="right", fontsize=9)
    ax.set_title(f"{model} — Δ vs nt-shuffle baseline")
    ax.set_ylabel("Δ score")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

fig.suptitle("How does each manipulation differ from nt-shuffle? (positive = easier)", fontsize=13)
fig.tight_layout()
fig.savefig(OUT / "manip_all_delta_vs_nt.png", dpi=150)
plt.close(fig)


print("Saved comparison charts to:", OUT)
for f in sorted(OUT.glob("manip_all_*.png")):
    print(" ", f.name)
