"""
SpliceMamba v3 architecture diagram.

Renders a vertical flow chart of the forward pass with module names,
hyperparameters, and tensor shapes at each stage. Saves PNG + PDF to
evaluation/results/architecture.png|.pdf.

Declarative only — does not import torch / mamba-ssm / flash-attn.
Run: python visualize_architecture.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


# ---------------------------------------------------------------------------
# Palette (muted, print-friendly)
# ---------------------------------------------------------------------------
COLORS = {
    "io":      ("#E8E8E8", "#4A4A4A"),  # fill, edge
    "stem":    ("#CFE2F3", "#1F4E79"),
    "posenc":  ("#C9E7E4", "#1D6F6A"),
    "mamba":   ("#DBCCEB", "#5B3A82"),
    "coarse":  ("#FCE5CD", "#B45309"),
    "attn":    ("#D9EAD3", "#2F6F2F"),
    "refined": ("#F4CCCC", "#9B1C1C"),
    "callout": ("#FFFFFF", "#999999"),
}

SHAPE_FILL = "#FFF8E1"
SHAPE_EDGE = "#8A6D00"


# ---------------------------------------------------------------------------
# Drawing primitives
# ---------------------------------------------------------------------------
def draw_box(
    ax,
    x, y, w, h,
    title, subtitle=None,
    color_key="stem",
    title_size=11,
    sub_size=8.5,
):
    fill, edge = COLORS[color_key]
    patch = FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.15",
        linewidth=1.4, facecolor=fill, edgecolor=edge,
    )
    ax.add_patch(patch)
    if subtitle:
        # place title in upper third, subtitle in lower two-thirds
        ax.text(x, y + h * 0.22, title, ha="center", va="center",
                fontsize=title_size, fontweight="bold", color=edge)
        ax.text(x, y - h * 0.18, subtitle, ha="center", va="center",
                fontsize=sub_size, color="#333333")
    else:
        ax.text(x, y, title, ha="center", va="center",
                fontsize=title_size, fontweight="bold", color=edge)


def draw_shape_tag(ax, x, y, shape_text, w=3.4, h=0.48):
    patch = FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.1",
        linewidth=1.0, facecolor=SHAPE_FILL, edgecolor=SHAPE_EDGE,
    )
    ax.add_patch(patch)
    ax.text(x, y, shape_text, ha="center", va="center",
            fontsize=9, family="monospace", color="#5A4300")


def draw_arrow(ax, x1, y1, x2, y2, color="#333333", lw=1.6, style="-|>"):
    arrow = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle=style, mutation_scale=14,
        linewidth=lw, color=color, shrinkA=0, shrinkB=0,
    )
    ax.add_patch(arrow)


def draw_callout(ax, x, y, w, h, title, lines, color_key="callout"):
    fill, edge = COLORS[color_key]
    patch = FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0.03,rounding_size=0.1",
        linewidth=1.0, facecolor=fill, edgecolor=edge,
        linestyle="--",
    )
    ax.add_patch(patch)
    ax.text(x, y + h / 2 - 0.28, title, ha="center", va="top",
            fontsize=9.5, fontweight="bold", color=edge)
    # evenly space lines
    inner_top = y + h / 2 - 0.68
    inner_bot = y - h / 2 + 0.18
    n = max(len(lines), 1)
    step = (inner_top - inner_bot) / max(n - 1, 1) if n > 1 else 0
    for i, line in enumerate(lines):
        ty = inner_top - i * step
        ax.text(x, ty, line, ha="center", va="center",
                fontsize=8.3, color="#333333", family="monospace")


# ---------------------------------------------------------------------------
# Architecture (declarative)
# ---------------------------------------------------------------------------
STAGES = [
    ("io",      "Input  one-hot DNA",
                 "A/C/G/T channels · 15 kb window",
                 "(B, 4, 15000)"),
    ("stem",    "DeepConvStem · input_conv",
                 "Conv1d(4→256, k=11, pad=5) + BN + GELU",
                 "(B, 256, 15000)"),
    ("stem",    "DeepConvStem · 4× DilatedResBlock",
                 "k=5 · dilations [1, 4, 10, 25] · RF ≈ 641 bp",
                 "(B, 256, 15000)"),
    ("stem",    "Transpose  (B, D, L) → (B, L, D)",
                 None,
                 "(B, 15000, 256)"),
    ("posenc",  "SinusoidalPositionalEncoding",
                 "additive · max_len=15000 · d=256",
                 "(B, 15000, 256)"),
    ("mamba",   "BiMambaEncoder   (8 fwd + 8 bwd)",
                 "Mamba2: d_state=64, expand=2, d_conv=4, headdim=32\n"
                 "fusion = Linear(512→256)",
                 "(B, 15000, 256)"),
]

ATTN_STAGE = (
    "attn",
    "SlidingWindowAttention × 4",
    "FlashAttn-2 · 8 heads · window_radius=400\nFFN expand ×4",
    "(B, 15000, 256)",
)

REFINED_STAGE = (
    "refined",
    "refined_head   ClassificationHead",
    "Linear(256→256) → GELU → Dropout → Linear(256→3)",
    "(B, 15000, 3)",
)

COARSE_STAGE = (
    "coarse",
    "coarse_head   ClassificationHead",
    "aux loss (λ=0.1)\nLinear→GELU→Dropout→Linear",
    "(B, 15000, 3)",
)


# ---------------------------------------------------------------------------
# Main figure
# ---------------------------------------------------------------------------
def build_figure():
    fig, ax = plt.subplots(figsize=(15.5, 19))
    ax.set_xlim(0, 17)
    ax.set_ylim(0, 24)
    ax.set_aspect("equal")
    ax.axis("off")

    spine_x = 7.2
    box_w = 6.4
    box_h = 1.35

    # ------------------------------------------------------------------
    # Title
    # ------------------------------------------------------------------
    ax.text(spine_x, 23.3, "SpliceMamba v3 — Forward Pass",
            ha="center", va="center", fontsize=18, fontweight="bold",
            color="#222222")
    ax.text(spine_x, 22.65,
            "defaults: d_model=256, n_mamba_layers=8, n_attn_layers=4, "
            "window_radius=400",
            ha="center", va="center", fontsize=10.5, color="#555555",
            style="italic")

    # ------------------------------------------------------------------
    # Main vertical spine
    # ------------------------------------------------------------------
    # y-coordinates (top → bottom) for the 6 pre-branch stages
    ys = [21.3, 19.7, 18.1, 16.7, 15.3, 13.5]
    assert len(ys) == len(STAGES)

    for (color, title, sub, shape), y in zip(STAGES, ys):
        h = box_h + 0.45 if color == "mamba" else box_h
        draw_box(ax, spine_x, y, box_w, h, title, sub, color_key=color)

    # Shape tags between main-column stages
    for i in range(len(STAGES) - 1):
        y_between = (ys[i] + ys[i + 1]) / 2
        # account for taller mamba box
        top_half = box_h / 2
        if STAGES[i][0] == "mamba":
            top_half = (box_h + 0.45) / 2
        draw_shape_tag(ax, spine_x, y_between, STAGES[i][3])
        draw_arrow(ax, spine_x, ys[i] - top_half - 0.02,
                   spine_x, y_between + 0.27)
        bot_half = box_h / 2
        if STAGES[i + 1][0] == "mamba":
            bot_half = (box_h + 0.45) / 2
        draw_arrow(ax, spine_x, y_between - 0.27,
                   spine_x, ys[i + 1] + bot_half + 0.02)

    # encoder_output shape tag below mamba
    mamba_bot = ys[-1] - (box_h + 0.45) / 2
    y_branch = mamba_bot - 0.75
    draw_shape_tag(ax, spine_x, y_branch,
                   STAGES[-1][3] + "   ← encoder_output", w=5.8)
    draw_arrow(ax, spine_x, mamba_bot - 0.02, spine_x, y_branch + 0.27)

    # ------------------------------------------------------------------
    # Coarse head branch (right side)
    # ------------------------------------------------------------------
    coarse_x = 13.6
    coarse_y = y_branch - 1.4
    draw_box(ax, coarse_x, coarse_y, 5.4, box_h + 0.15,
             COARSE_STAGE[1], COARSE_STAGE[2], color_key="coarse",
             title_size=10.5, sub_size=8.2)

    # shape tag + terminal box
    draw_shape_tag(ax, coarse_x, coarse_y - 1.3, COARSE_STAGE[3], w=3.0)
    coarse_term_y = coarse_y - 2.55
    draw_box(ax, coarse_x, coarse_term_y, 3.8, 0.8,
             "coarse_logits", None, color_key="io", title_size=10)

    # elbow arrow from shape tag at spine → coarse
    ax.plot(
        [spine_x + 2.9, coarse_x - 2.75, coarse_x - 2.75],
        [y_branch, y_branch, coarse_y + 0.3],
        color="#B45309", lw=1.8, solid_capstyle="round",
    )
    draw_arrow(ax, coarse_x - 2.75, coarse_y + 0.35,
               coarse_x - 2.72, coarse_y + 0.0,
               color="#B45309", lw=1.8)
    draw_arrow(ax, coarse_x, coarse_y - (box_h + 0.15) / 2 - 0.02,
               coarse_x, coarse_y - 1.08)
    draw_arrow(ax, coarse_x, coarse_y - 1.55,
               coarse_x, coarse_term_y + 0.42)

    # ------------------------------------------------------------------
    # Attention + refined (down the spine)
    # ------------------------------------------------------------------
    attn_y = y_branch - 1.8
    draw_box(ax, spine_x, attn_y, box_w, box_h + 0.4,
             ATTN_STAGE[1], ATTN_STAGE[2], color_key="attn")
    draw_arrow(ax, spine_x, y_branch - 0.27,
               spine_x, attn_y + (box_h + 0.4) / 2 + 0.02)

    shape_after_attn_y = attn_y - (box_h + 0.4) / 2 - 0.55
    draw_shape_tag(ax, spine_x, shape_after_attn_y, ATTN_STAGE[3])
    draw_arrow(ax, spine_x, attn_y - (box_h + 0.4) / 2 - 0.02,
               spine_x, shape_after_attn_y + 0.27)

    refined_y = shape_after_attn_y - 1.45
    draw_box(ax, spine_x, refined_y, box_w, box_h + 0.15,
             REFINED_STAGE[1], REFINED_STAGE[2], color_key="refined")
    draw_arrow(ax, spine_x, shape_after_attn_y - 0.27,
               spine_x, refined_y + (box_h + 0.15) / 2 + 0.02)

    final_shape_y = refined_y - (box_h + 0.15) / 2 - 0.55
    draw_shape_tag(ax, spine_x, final_shape_y, REFINED_STAGE[3])
    draw_arrow(ax, spine_x, refined_y - (box_h + 0.15) / 2 - 0.02,
               spine_x, final_shape_y + 0.27)

    final_y = final_shape_y - 0.85
    draw_box(ax, spine_x, final_y, 4.2, 0.8,
             "refined_logits  (primary loss)", None,
             color_key="io", title_size=10.5)
    draw_arrow(ax, spine_x, final_shape_y - 0.27,
               spine_x, final_y + 0.42)

    # Label-region callout on the refined output
    ann_text = ("loss & metrics use\n"
                "positions [5000 : 10000]\n"
                "(central 5 kb label region)")
    ax.annotate(
        ann_text,
        xy=(spine_x + 2.1, final_y),
        xytext=(spine_x + 4.3, final_y + 0.4),
        fontsize=8.8, color="#7A1C1C",
        ha="left", va="center",
        arrowprops=dict(arrowstyle="->", color="#7A1C1C", lw=1.0),
        bbox=dict(boxstyle="round,pad=0.35",
                  fc="#FDF2F2", ec="#9B1C1C", lw=0.8),
    )

    # ------------------------------------------------------------------
    # Left-side callouts: inner structure of the three key block types
    # ------------------------------------------------------------------
    draw_callout(
        ax, x=2.05, y=18.1, w=3.5, h=2.5,
        title="DilatedResBlock",
        lines=[
            "x ──┐",
            "Conv · BN · GELU",
            "Conv · BN",
            "(+ x)  → GELU",
        ],
    )
    draw_callout(
        ax, x=2.05, y=13.5, w=3.5, h=2.9,
        title="Mamba2Layer  (×8 per direction)",
        lines=[
            "x ──┐",
            "LayerNorm",
            "Mamba2",
            "Dropout · DropPath",
            "(+ x)",
        ],
    )
    draw_callout(
        ax, x=2.05, y=attn_y, w=3.5, h=3.3,
        title="SlidingWindowAttention",
        lines=[
            "x ──┐",
            "LN · Q/K/V proj",
            "flash_attn(window=400)",
            "out_proj · DP   (+ x)",
            "LN · FFN(×4) · DP",
            "(+ x)",
        ],
    )

    # Right-side BiMamba fusion inset
    draw_callout(
        ax, x=13.6, y=13.5, w=5.2, h=2.5,
        title="BiMambaEncoder — fusion",
        lines=[
            "h_fwd = stack(x)",
            "h_bwd = stack(x.flip(1)).flip(1)",
            "h = Linear_{2D→D}(",
            "      cat[h_fwd, h_bwd])",
        ],
    )

    # ------------------------------------------------------------------
    # Legend + glossary footer
    # ------------------------------------------------------------------
    legend_y = 0.9
    legend_items = [
        ("stem",    "Conv stem"),
        ("posenc",  "Positional enc"),
        ("mamba",   "BiMamba encoder"),
        ("coarse",  "Coarse head (aux)"),
        ("attn",    "Sliding attn"),
        ("refined", "Refined head"),
    ]
    x0 = 1.2
    dx = 2.55
    for i, (key, label) in enumerate(legend_items):
        fill, edge = COLORS[key]
        x = x0 + i * dx
        patch = FancyBboxPatch(
            (x, legend_y), 0.5, 0.32,
            boxstyle="round,pad=0.01,rounding_size=0.06",
            facecolor=fill, edgecolor=edge, lw=1.0,
        )
        ax.add_patch(patch)
        ax.text(x + 0.62, legend_y + 0.16, label,
                fontsize=9, va="center", color="#333")

    ax.text(
        spine_x + 0.5, 0.3,
        "B = batch · L = sequence length (15 000) · D = d_model (256) · "
        "3 classes = {neither, acceptor, donor}",
        ha="center", va="center", fontsize=9, color="#555555",
        style="italic",
    )

    return fig


# ---------------------------------------------------------------------------
# Compact 4-panel comparison: SpliceAI vs SpliceMamba v2 / v3 / big-v4
# ---------------------------------------------------------------------------
# Each panel: a single vertical column of color-coded blocks with one-line
# hyperparameter subtitle. Designed to fit side-by-side at slide resolution.

# (color_key, title, subtitle)  — one block per stage, top → bottom.
PANELS = {
    "SpliceAI (Jaganathan 2019)": [
        ("io",      "Input one-hot DNA",            "(B, 4, L)"),
        ("stem",    "Conv1d(4→32, k=1)",            "input projection"),
        ("stem",    "32× DilatedResBlock",          "dilations 1·4·10·25\nRF ≈ 10 kb"),
        ("posenc",  "Skip-connection sum",          "every-4-block skip"),
        ("refined", "Conv1d(32→3)",                 "per-base 3-class logits"),
        ("io",      "Output  logits",               "(B, L, 3)"),
    ],
    "SpliceMamba v2 (CE)": [
        ("io",      "Input one-hot DNA",            "(B, 4, L)"),
        ("stem",    "Shallow ConvStem",             "Conv1d(4→256, k=7) + GELU"),
        ("posenc",  "Sinusoidal PE",                "additive · max_len=15000"),
        ("mamba",   "BiMambaEncoder",               "6 layers · d=256 · d_state=64"),
        ("refined", "ClassificationHead",           "single head · CE loss"),
        ("io",      "Output  logits",               "(B, L, 3)"),
    ],
    "SpliceMamba v3 (ensemble)": [
        ("io",      "Input one-hot DNA",            "(B, 4, L)"),
        ("stem",    "DeepConvStem",                 "4× DilatedResBlock\nd 1·4·10·25 · RF ≈ 641 bp"),
        ("posenc",  "Sinusoidal PE",                "additive"),
        ("mamba",   "BiMambaEncoder",               "8 layers · d=256 · d_state=64"),
        ("coarse",  "coarse_head (aux λ=0.1)",      "auxiliary supervision"),
        ("attn",    "SlidingWindowAttn × 4",        "FlashAttn-2 · R=400 · 8 heads"),
        ("refined", "refined_head",                 "primary loss · 5-model ensemble"),
        ("io",      "Output  logits",               "(B, L, 3)"),
    ],
    "SpliceMamba big (v4) — best": [
        ("io",      "Input one-hot DNA",            "(B, 4, L)"),
        ("stem",    "DeepConvStem",                 "4× DilatedResBlock\nd 1·4·10·25 · RF ≈ 641 bp"),
        ("posenc",  "Sinusoidal PE",                "additive"),
        ("mamba",   "BiMambaEncoder",               "8 layers · d=512 · d_state=64"),
        ("coarse",  "coarse_head (aux λ=0.1)",      "auxiliary supervision"),
        ("attn",    "SlidingWindowAttn × 8",        "FlashAttn-2 · R=200 · 8 heads"),
        ("refined", "refined_head",                 "primary loss · label smooth 0.05"),
        ("io",      "Output  logits",               "(B, L, 3)"),
    ],
}


def build_comparison_figure():
    """4-panel side-by-side comparison of architectures."""
    titles = list(PANELS.keys())
    n_panels = len(titles)

    # max blocks across panels controls vertical span
    max_blocks = max(len(blocks) for blocks in PANELS.values())

    panel_w = 5.0       # width per panel
    fig_w = panel_w * n_panels
    block_h = 1.05
    block_w = 4.2
    v_gap = 0.45        # vertical gap between blocks (centers)
    title_h = 1.6       # space reserved at top for title

    fig_h = title_h + max_blocks * (block_h + v_gap) + 0.8
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(0, fig_w)
    ax.set_ylim(0, fig_h)
    ax.set_aspect("equal")
    ax.axis("off")

    fig.suptitle(
        "Architecture comparison — SpliceAI vs SpliceMamba variants",
        fontsize=15, fontweight="bold", y=0.995, color="#222222",
    )

    for p_idx, title in enumerate(titles):
        blocks = PANELS[title]
        x_center = panel_w * p_idx + panel_w / 2

        # panel title
        ax.text(
            x_center, fig_h - 0.6, title,
            ha="center", va="center",
            fontsize=12, fontweight="bold", color="#222222",
        )

        # blocks (top → bottom)
        y_top = fig_h - title_h
        for b_idx, (color_key, btitle, subtitle) in enumerate(blocks):
            y = y_top - b_idx * (block_h + v_gap) - block_h / 2
            h = block_h + (0.35 if subtitle and "\n" in (subtitle or "") else 0.0)
            draw_box(
                ax, x_center, y, block_w, h,
                btitle, subtitle, color_key=color_key,
                title_size=10.5, sub_size=8.2,
            )
            # arrow to next
            if b_idx < len(blocks) - 1:
                y_next = y_top - (b_idx + 1) * (block_h + v_gap) - block_h / 2
                draw_arrow(
                    ax,
                    x_center, y - h / 2 - 0.02,
                    x_center, y_next + block_h / 2 + 0.02,
                    lw=1.3,
                )

        # parameter-count footer
        param_str = {
            "SpliceAI (Jaganathan 2019)":      "~340 K params · pure conv · single head",
            "SpliceMamba v2 (CE)":             "~6 M params · SSM · single head",
            "SpliceMamba v3 (ensemble)":       "~12 M × 5 · SSM + local attn · dual head",
            "SpliceMamba big (v4) — best":     "~30 M params · SSM + local attn · dual head",
        }[title]
        ax.text(
            x_center, 0.45, param_str,
            ha="center", va="center", fontsize=9, color="#555555",
            style="italic",
        )

    return fig


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument(
        "--variant", default="v3",
        choices=["v3", "compare", "all"],
        help="v3: detailed v3 diagram (default). "
             "compare: 4-panel SpliceAI/v2/v3/big. "
             "all: emit both.",
    )
    args = p.parse_args()

    repo_root = Path(__file__).parent
    out_v3 = repo_root / "evaluation" / "results"
    out_cmp = repo_root / "evaluation" / "results-bigmodel"
    out_v3.mkdir(parents=True, exist_ok=True)
    out_cmp.mkdir(parents=True, exist_ok=True)

    if args.variant in ("v3", "all"):
        fig = build_figure()
        fig.savefig(out_v3 / "architecture.png", dpi=200,
                    bbox_inches="tight", facecolor="white")
        fig.savefig(out_v3 / "architecture.pdf",
                    bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"saved  {out_v3 / 'architecture.png'}")

    if args.variant in ("compare", "all"):
        fig = build_comparison_figure()
        fig.savefig(out_cmp / "architectures_compared.png", dpi=200,
                    bbox_inches="tight", facecolor="white")
        fig.savefig(out_cmp / "architectures_compared.pdf",
                    bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"saved  {out_cmp / 'architectures_compared.png'}")


if __name__ == "__main__":
    main()
