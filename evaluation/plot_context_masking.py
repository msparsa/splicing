"""
Aggregate context-masking sweep predictions and produce the final plot.

Reads:   evaluation/results/behavior/context_masking_{model}_R{radius}.npz
Writes:  evaluation/results/behavior/context_masking_sweep.png
         evaluation/results/behavior/context_masking_summary.json

For each (model, radius) pair, computes:
  * AUPRC (donor, acceptor)
  * Top-k global accuracy (donor, acceptor)
  * Recall @ 0.5 (donor, acceptor)

Radii are interpreted as per-side distance from window center (7500).
R=5000 is the unmasked 15kb input (baseline).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
_EVAL_DIR = _REPO_ROOT / "evaluation"
if str(_EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(_EVAL_DIR))

from eval_utils import (  # noqa: E402
    compute_gene_window_counts,
    read_window_labels,
    stitch_gene_labels,
    stitch_gene_predictions,
    compute_auprc,
    compute_topk_accuracy,
    make_serializable,
)

DATASET_H5 = _REPO_ROOT / "dataset_test_0.h5"
DATAFILE_H5 = _REPO_ROOT / "datafile_test_0.h5"
BEHAVIOR_DIR = _EVAL_DIR / "results" / "behavior"


def _metrics_from_probs(probs: np.ndarray, labels_flat: np.ndarray,
                        windows_per_gene: np.ndarray) -> dict:
    """Compute donor/acceptor AUPRC, global top-k, recall@0.5 for one (model,R)."""
    gene_probs = stitch_gene_predictions(probs, windows_per_gene)
    gene_labels = stitch_gene_labels(
        labels_flat.reshape(-1, 5000), windows_per_gene)

    auprc = compute_auprc(gene_probs, gene_labels)
    topk = compute_topk_accuracy(gene_probs, gene_labels)

    # Recall @ 0.5 on flat arrays
    all_probs = probs.reshape(-1, probs.shape[-1])
    all_labels = labels_flat.reshape(-1)
    out = {
        "auprc_donor": float(auprc["auprc_donor"]),
        "auprc_acceptor": float(auprc["auprc_acceptor"]),
        "topk_global_donor": float(topk["topk_global_donor"]),
        "topk_global_acceptor": float(topk["topk_global_acceptor"]),
    }
    for cls_name, cls_idx in [("acceptor", 1), ("donor", 2)]:
        m = all_labels == cls_idx
        if m.any():
            out[f"recall@0.5_{cls_name}"] = float(
                (all_probs[m, cls_idx] >= 0.5).mean()
            )
        else:
            out[f"recall@0.5_{cls_name}"] = 0.0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--radii", type=int, nargs="+",
                    default=[500, 1000, 2500, 5000])
    ap.add_argument("--out-dir", default=str(BEHAVIOR_DIR))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    labels_flat = read_window_labels(str(DATASET_H5))
    windows_per_gene = compute_gene_window_counts(str(DATAFILE_H5))

    summary = {}  # {model: {radius: metrics}}

    for model in ("splicemamba", "spliceai"):
        summary[model] = {}
        for R in args.radii:
            path = out_dir / f"context_masking_{model}_R{R}.npz"
            if not path.exists():
                print(f"[warn] missing {path.name} — skipping", file=sys.stderr)
                continue
            print(f"[load] {path.name} …", file=sys.stderr)
            probs = np.load(path)["probs"]
            metrics = _metrics_from_probs(probs, labels_flat, windows_per_gene)
            print(f"[metrics] {model} R={R}: "
                  f"AUPRC donor={metrics['auprc_donor']:.4f} "
                  f"acceptor={metrics['auprc_acceptor']:.4f} "
                  f"topk d={metrics['topk_global_donor']:.4f} "
                  f"a={metrics['topk_global_acceptor']:.4f}",
                  file=sys.stderr)
            summary[model][R] = metrics

    # --- Plot: AUPRC (d/a) and top-k (d/a) vs radius for both models
    colors = {"splicemamba": "tab:blue", "spliceai": "tab:orange"}
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=True)
    metric_positions = [
        ("auprc_donor", "AUPRC — donor", axes[0, 0]),
        ("auprc_acceptor", "AUPRC — acceptor", axes[0, 1]),
        ("topk_global_donor", "Top-k (global) — donor", axes[1, 0]),
        ("topk_global_acceptor", "Top-k (global) — acceptor", axes[1, 1]),
    ]
    for metric_key, title, ax in metric_positions:
        for model, model_data in summary.items():
            if not model_data:
                continue
            radii = sorted(model_data.keys())
            vals = [model_data[r][metric_key] for r in radii]
            ax.plot(radii, vals, "o-", color=colors[model], label=model,
                    linewidth=2, markersize=7)
        ax.set_title(title)
        ax.set_xlabel("Per-side radius (bp) around window center")
        ax.set_ylabel(metric_key)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="lower right")
        ax.set_xscale("log")

    fig.suptitle(
        "Context masking sweep: model performance vs available input context "
        "(R=5000 is full 15kb input)",
        fontsize=11,
    )
    fig.tight_layout()
    out_path = out_dir / "context_masking_sweep.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[done] wrote {out_path}", file=sys.stderr)

    summary_path = out_dir / "context_masking_summary.json"
    with open(summary_path, "w") as f:
        json.dump(make_serializable(summary), f, indent=2)
    print(f"[done] wrote {summary_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
