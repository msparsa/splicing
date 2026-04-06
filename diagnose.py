"""
SpliceMamba diagnostic analysis.

Runs three analyses on the test set to diagnose model shortcomings:
  1. Is the attention helping? (coarse vs refined comparison)
  2. Where are the errors? (FN categorization by structural features)
  3. Is the gate working? (gate value distribution analysis)

Usage:
    python diagnose.py --checkpoint checkpoints/best.pt --output diagnostics/
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import average_precision_score

from evaluate import (
    EVAL_CONFIG,
    load_model,
    compute_gene_window_counts,
    stitch_gene_predictions,
    stitch_gene_labels,
    read_window_labels,
    compute_auprc,
    compute_topk_accuracy,
)


# ---------------------------------------------------------------------------
# Modified inference: captures coarse, refined, and gate values
# ---------------------------------------------------------------------------

@torch.no_grad()
def predict_windows_diagnostic(
    model: torch.nn.Module,
    dataset_path: str,
    cfg: dict,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Run inference on all test windows, capturing coarse and refined outputs.

    Returns
    -------
    coarse_probs  : (total_windows, 5000, 3) float32
    refined_probs : (total_windows, 5000, 3) float32
    """
    with h5py.File(dataset_path, "r") as f:
        x_keys = sorted(
            [k for k in f.keys() if k.startswith("X")],
            key=lambda k: int(k[1:]),
        )
        all_coarse, all_refined = [], []

        for x_key in x_keys:
            shard_num = int(x_key[1:])
            x_data = f[x_key][:]  # (N, 15000, 4) int8
            n_windows = x_data.shape[0]
            print(f"  Shard {x_key}: {n_windows} windows")

            for start in range(0, n_windows, cfg["batch_size"]):
                end = min(start + cfg["batch_size"], n_windows)
                batch = torch.from_numpy(
                    x_data[start:end].astype(np.float32)
                ).permute(0, 2, 1).to(device)  # (B, 4, 15000)

                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    coarse_logits, refined_logits, _ = model(batch)

                ls, le = cfg["label_start"], cfg["label_end"]

                # Coarse probs
                coarse_label = coarse_logits[:, ls:le, :]
                coarse_probs = torch.softmax(coarse_label.float(), dim=-1)

                # Refined probs
                refined_label = refined_logits[:, ls:le, :]
                refined_probs = torch.softmax(refined_label.float(), dim=-1)

                all_coarse.append(coarse_probs.cpu().numpy())
                all_refined.append(refined_probs.cpu().numpy())

    return (
        np.concatenate(all_coarse, axis=0),
        np.concatenate(all_refined, axis=0),
    )


# ---------------------------------------------------------------------------
# Diagnosis 1: Is attention helping?
# ---------------------------------------------------------------------------

def run_diagnosis_1(
    gene_coarse: list[np.ndarray],
    gene_refined: list[np.ndarray],
    gene_labels: list[np.ndarray],
) -> dict:
    """Compare coarse (Mamba-only) vs refined (Mamba + attention) metrics."""
    print("\n" + "=" * 60)
    print("DIAGNOSIS 1: Is Attention Helping?")
    print("=" * 60)

    auprc_coarse = compute_auprc(gene_coarse, gene_labels)
    auprc_refined = compute_auprc(gene_refined, gene_labels)
    topk_coarse = compute_topk_accuracy(gene_coarse, gene_labels)
    topk_refined = compute_topk_accuracy(gene_refined, gene_labels)

    results = {
        "coarse": {**auprc_coarse, **topk_coarse},
        "refined": {**auprc_refined, **topk_refined},
        "delta": {},
    }

    # Print comparison table
    print(f"\n{'Metric':<30} {'Coarse':>10} {'Refined':>10} {'Delta':>10}")
    print("-" * 62)
    for key in ["auprc_donor", "auprc_acceptor", "auprc_mean",
                "topk_global_donor", "topk_global_acceptor", "topk_global_mean"]:
        c = auprc_coarse.get(key, topk_coarse.get(key, 0.0))
        r = auprc_refined.get(key, topk_refined.get(key, 0.0))
        d = r - c
        results["delta"][key] = d
        print(f"  {key:<28} {c:>10.4f} {r:>10.4f} {d:>+10.4f}")

    auprc_gap = results["delta"]["auprc_mean"]
    topk_gap = results["delta"]["topk_global_mean"]
    print(f"\nAUPRC mean gap: {auprc_gap:+.4f}")
    print(f"Top-k global mean gap: {topk_gap:+.4f}")

    if abs(auprc_gap) < 0.03 and abs(topk_gap) < 0.03:
        print("\n** ATTENTION IS CONTRIBUTING LITTLE (<3% gap) **")
        print("   -> Consider: deepen attention (2->4 layers), widen window (R=200->400)")
    else:
        print(f"\n   Attention contributes {auprc_gap:+.1%} AUPRC, {topk_gap:+.1%} top-k")

    return results


# ---------------------------------------------------------------------------
# Diagnosis 2: Where are the errors?
# ---------------------------------------------------------------------------

def parse_gene_junctions(datafile_path: str) -> list[dict]:
    """Parse junction and sequence data for all test genes.

    Returns a list of dicts, one per gene, with keys:
        tx_start, tx_end, donors, acceptors, seq,
        intron_lengths, exon_lengths,
        donor_dinucs, acceptor_dinucs
    """
    genes = []
    with h5py.File(datafile_path, "r") as f:
        n_genes = f["TX_START"].shape[0]
        for g in range(n_genes):
            tx_start = int(f["TX_START"][g])
            tx_end = int(f["TX_END"][g])

            # Parse sequence
            seq_raw = f["SEQ"][g]
            if isinstance(seq_raw, bytes):
                seq = seq_raw.rstrip(b"\x00").decode("ascii", errors="replace")
            else:
                seq = str(seq_raw).rstrip("\x00")

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
                    # This exon is bounded by acceptor[i] and donor[i+1]
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

            # Read strand
            strand_raw = f["STRAND"][g]
            if isinstance(strand_raw, bytes):
                strand = strand_raw.decode()
            else:
                strand = str(strand_raw)

            # Determine canonical status per intron
            # JN_START+1 has the first 2 bases of the intron on the genomic strand
            # JN_END-2 to JN_END has the last 2 bases of the intron on the genomic strand
            # + strand canonical: GT...AG
            # - strand canonical: CT...AC (reverse complement of GT...AG)
            intron_canonical = {}  # keyed by (donor_coord, acceptor_coord)
            for d, a in zip(donors, acceptors):
                # Donor-end dinucleotide (first 2 bases of intron)
                d_pos = d - tx_start + 5000 + 1
                if 0 <= d_pos < len(seq) - 1:
                    d_dinuc = seq[d_pos:d_pos + 2].upper()
                else:
                    d_dinuc = "??"

                # Acceptor-end dinucleotide (last 2 bases of intron)
                a_pos = a - tx_start + 5000
                if 2 <= a_pos <= len(seq):
                    a_dinuc = seq[a_pos - 2:a_pos].upper()
                else:
                    a_dinuc = "??"

                is_canonical = (
                    (d_dinuc == "GT" and a_dinuc == "AG") or  # + strand
                    (d_dinuc == "CT" and a_dinuc == "AC")     # - strand
                )
                intron_canonical[d] = is_canonical
                intron_canonical[a] = is_canonical

            genes.append({
                "tx_start": tx_start,
                "tx_end": tx_end,
                "strand": strand,
                "donors": donors,
                "acceptors": acceptors,
                "intron_lengths": intron_lengths,
                "exon_lengths": exon_lengths,
                "intron_canonical": intron_canonical,
            })

    return genes


def run_diagnosis_2(
    gene_refined: list[np.ndarray],
    gene_labels: list[np.ndarray],
    cfg: dict,
    threshold: float = 0.5,
) -> dict:
    """Categorize false negatives by structural features."""
    print("\n" + "=" * 60)
    print("DIAGNOSIS 2: Where Are the Errors?")
    print("=" * 60)

    print("Parsing gene junction data...")
    genes = parse_gene_junctions(cfg["test_datafile_path"])

    # Intron length buckets
    intron_buckets = {
        "<200bp": (0, 200),
        "200-1000bp": (200, 1000),
        "1000-5000bp": (1000, 5000),
        ">5000bp": (5000, float("inf")),
    }
    # Exon length buckets
    exon_buckets = {
        "<80bp": (0, 80),
        "80-200bp": (80, 200),
        "200-500bp": (200, 500),
        ">500bp": (500, float("inf")),
    }
    # Position buckets (quintiles of relative position in gene label array)
    position_buckets = {
        "0-20%": (0.0, 0.2),
        "20-40%": (0.2, 0.4),
        "40-60%": (0.4, 0.6),
        "60-80%": (0.6, 0.8),
        "80-100%": (0.8, 1.0),
    }

    # Initialize counters: {bucket_name: {"tp": 0, "fn": 0}}
    def make_counters(buckets):
        return {
            cls: {b: {"tp": 0, "fn": 0} for b in buckets}
            for cls in ["donor", "acceptor"]
        }

    intron_counts = make_counters(intron_buckets)
    exon_counts = make_counters(exon_buckets)
    position_counts = make_counters(position_buckets)
    dinuc_counts = {
        cls: {"canonical": {"tp": 0, "fn": 0}, "non-canonical": {"tp": 0, "fn": 0}}
        for cls in ["donor", "acceptor"]
    }

    total_tp = {"donor": 0, "acceptor": 0}
    total_fn = {"donor": 0, "acceptor": 0}

    for gene_idx, (probs, labels, gene_info) in enumerate(
        zip(gene_refined, gene_labels, genes)
    ):
        gene_len_labels = len(labels)
        tx_start = gene_info["tx_start"]

        for cls_name, cls_idx in [("acceptor", 1), ("donor", 2)]:
            true_positions = np.where(labels == cls_idx)[0]
            if len(true_positions) == 0:
                continue

            scores = probs[:, cls_idx]
            genomic_positions = (
                gene_info["donors"] if cls_name == "donor"
                else gene_info["acceptors"]
            )

            for pos in true_positions:
                is_tp = scores[pos] >= threshold
                status = "tp" if is_tp else "fn"

                if is_tp:
                    total_tp[cls_name] += 1
                else:
                    total_fn[cls_name] += 1

                # Map label position to genomic coordinate
                genomic_coord = pos + tx_start

                # Intron length bucketing
                intron_len = gene_info["intron_lengths"].get(genomic_coord)
                if intron_len is not None:
                    for bname, (lo, hi) in intron_buckets.items():
                        if lo <= intron_len < hi:
                            intron_counts[cls_name][bname][status] += 1
                            break

                # Exon length bucketing
                exon_len = gene_info["exon_lengths"].get(genomic_coord)
                if exon_len is not None:
                    for bname, (lo, hi) in exon_buckets.items():
                        if lo <= exon_len < hi:
                            exon_counts[cls_name][bname][status] += 1
                            break

                # Position bucketing (relative position in gene)
                rel_pos = pos / max(gene_len_labels, 1)
                for bname, (lo, hi) in position_buckets.items():
                    if lo <= rel_pos < hi or (hi == 1.0 and rel_pos == 1.0):
                        position_counts[cls_name][bname][status] += 1
                        break

                # Dinucleotide bucketing (strand-aware canonical check)
                is_canonical = gene_info["intron_canonical"].get(
                    genomic_coord, False
                )
                cat = "canonical" if is_canonical else "non-canonical"
                dinuc_counts[cls_name][cat][status] += 1

    # Build results and print tables
    results = {
        "threshold": threshold,
        "totals": {
            "donor_tp": total_tp["donor"], "donor_fn": total_fn["donor"],
            "acceptor_tp": total_tp["acceptor"], "acceptor_fn": total_fn["acceptor"],
        },
    }

    def print_table(title, counts, bucket_names):
        print(f"\n{title}:")
        print(f"  {'Bucket':<20} {'Donor FN rate':>15} {'(n)':>8} {'Acceptor FN rate':>18} {'(n)':>8}")
        print("  " + "-" * 72)
        table = {}
        for bname in bucket_names:
            row = {}
            for cls in ["donor", "acceptor"]:
                tp = counts[cls][bname]["tp"]
                fn = counts[cls][bname]["fn"]
                total = tp + fn
                rate = fn / total if total > 0 else 0.0
                row[f"{cls}_fn_rate"] = rate
                row[f"{cls}_n"] = total
            table[bname] = row
            print(f"  {bname:<20} {row['donor_fn_rate']:>14.1%} {row['donor_n']:>7d}"
                  f"  {row['acceptor_fn_rate']:>17.1%} {row['acceptor_n']:>7d}")
        return table

    results["by_intron_length"] = print_table(
        "By Intron Length", intron_counts, intron_buckets.keys()
    )
    results["by_exon_length"] = print_table(
        "By Exon Length", exon_counts, exon_buckets.keys()
    )
    results["by_position"] = print_table(
        "By Position in Label Region", position_counts, position_buckets.keys()
    )
    results["by_dinucleotide"] = print_table(
        "By Dinucleotide", dinuc_counts, ["canonical", "non-canonical"]
    )

    # Summary
    for cls in ["donor", "acceptor"]:
        total = total_tp[cls] + total_fn[cls]
        rate = total_fn[cls] / total if total > 0 else 0.0
        print(f"\n  {cls.capitalize()} overall: {total_fn[cls]} FN / {total} total = {rate:.1%} FN rate")

    return results


# ---------------------------------------------------------------------------
# Diagnosis 3: Is the gate working?
# ---------------------------------------------------------------------------

def run_diagnosis_3(
    gate_values: np.ndarray,
    all_labels: np.ndarray,
    windows_per_gene: np.ndarray,
) -> dict:
    """Analyze gate value distribution at splice sites vs non-splice."""
    print("\n" + "=" * 60)
    print("DIAGNOSIS 3: Is the Gate Working?")
    print("=" * 60)

    # Flatten gate and labels
    gate_flat = gate_values.reshape(-1)
    labels_flat = all_labels.reshape(-1)

    # Masks
    splice_mask = labels_flat > 0
    neither_mask = labels_flat == 0
    donor_mask = labels_flat == 2
    acceptor_mask = labels_flat == 1

    gate_splice = gate_flat[splice_mask]
    gate_neither = gate_flat[neither_mask]
    gate_donor = gate_flat[donor_mask]
    gate_acceptor = gate_flat[acceptor_mask]

    results = {}

    # Statistics
    for name, arr in [("splice", gate_splice), ("neither", gate_neither),
                      ("donor", gate_donor), ("acceptor", gate_acceptor)]:
        results[f"gate_{name}_mean"] = float(arr.mean())
        results[f"gate_{name}_std"] = float(arr.std())
        results[f"gate_{name}_median"] = float(np.median(arr))
        results[f"gate_{name}_min"] = float(arr.min())
        results[f"gate_{name}_max"] = float(arr.max())

    # Fraction above thresholds for splice sites
    for thresh in [0.5, 0.7, 0.9]:
        results[f"gate_splice_frac_above_{thresh}"] = float(
            (gate_splice > thresh).mean()
        )

    # Fraction below thresholds for non-splice
    for thresh in [0.1, 0.3]:
        results[f"gate_neither_frac_below_{thresh}"] = float(
            (gate_neither < thresh).mean()
        )

    # d-prime separation metric
    var_splice = gate_splice.var()
    var_neither = gate_neither.var()
    pooled_std = np.sqrt(0.5 * (var_splice + var_neither))
    d_prime = (gate_splice.mean() - gate_neither.mean()) / max(pooled_std, 1e-8)
    results["gate_d_prime"] = float(d_prime)

    # Print
    print(f"\n{'':>25} {'Splice':>10} {'Neither':>10} {'Donor':>10} {'Acceptor':>10}")
    print("  " + "-" * 65)
    print(f"  {'Mean':<23} {results['gate_splice_mean']:>10.4f} {results['gate_neither_mean']:>10.4f}"
          f" {results['gate_donor_mean']:>10.4f} {results['gate_acceptor_mean']:>10.4f}")
    print(f"  {'Std':<23} {results['gate_splice_std']:>10.4f} {results['gate_neither_std']:>10.4f}"
          f" {results['gate_donor_std']:>10.4f} {results['gate_acceptor_std']:>10.4f}")
    print(f"  {'Median':<23} {results['gate_splice_median']:>10.4f} {results['gate_neither_median']:>10.4f}"
          f" {results['gate_donor_median']:>10.4f} {results['gate_acceptor_median']:>10.4f}")

    print(f"\n  Splice gate > 0.5: {results['gate_splice_frac_above_0.5']:.1%}")
    print(f"  Splice gate > 0.7: {results['gate_splice_frac_above_0.7']:.1%}")
    print(f"  Splice gate > 0.9: {results['gate_splice_frac_above_0.9']:.1%}")
    print(f"  Neither gate < 0.1: {results['gate_neither_frac_below_0.1']:.1%}")
    print(f"  Neither gate < 0.3: {results['gate_neither_frac_below_0.3']:.1%}")
    print(f"\n  d-prime (splice vs neither): {d_prime:.3f}")

    if d_prime < 0.5:
        print("\n  ** GATE IS POORLY SEPARATING SPLICE FROM NON-SPLICE (d' < 0.5) **")
        print("     -> Coarse head not confident enough, or tau annealing too fast")

    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_diagnosis_1(results: dict, output_dir: Path):
    """Grouped bar chart: coarse vs refined for key metrics."""
    metrics = ["auprc_donor", "auprc_acceptor", "auprc_mean",
               "topk_global_donor", "topk_global_acceptor", "topk_global_mean"]
    labels = ["AUPRC\nDonor", "AUPRC\nAcceptor", "AUPRC\nMean",
              "Top-k\nDonor", "Top-k\nAcceptor", "Top-k\nMean"]

    coarse_vals = [results["coarse"][m] for m in metrics]
    refined_vals = [results["refined"][m] for m in metrics]

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    bars1 = ax.bar(x - width / 2, coarse_vals, width, label="Coarse (Mamba only)",
                   color="#5B9BD5", edgecolor="black", linewidth=0.5)
    bars2 = ax.bar(x + width / 2, refined_vals, width, label="Refined (Mamba + Attn)",
                   color="#ED7D31", edgecolor="black", linewidth=0.5)

    ax.set_ylabel("Score")
    ax.set_title("Diagnosis 1: Coarse vs Refined Head Performance")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.set_ylim(0.8, 1.0)
    ax.grid(axis="y", alpha=0.3)

    # Add value labels on bars
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=7)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=7)

    plt.tight_layout()
    plt.savefig(output_dir / "diagnosis1_coarse_vs_refined.png", dpi=150)
    plt.close()


def plot_diagnosis_2(results: dict, output_dir: Path):
    """Bar plots for FN rate by each structural feature."""
    categories = [
        ("by_intron_length", "Intron Length", "diagnosis2_fn_by_intron_length.png"),
        ("by_exon_length", "Exon Length", "diagnosis2_fn_by_exon_length.png"),
        ("by_position", "Position in Label Region", "diagnosis2_fn_by_position.png"),
        ("by_dinucleotide", "Dinucleotide", "diagnosis2_fn_by_dinucleotide.png"),
    ]

    for key, title, filename in categories:
        data = results[key]
        bucket_names = list(data.keys())

        donor_rates = [data[b]["donor_fn_rate"] for b in bucket_names]
        acceptor_rates = [data[b]["acceptor_fn_rate"] for b in bucket_names]
        donor_n = [data[b]["donor_n"] for b in bucket_names]
        acceptor_n = [data[b]["acceptor_n"] for b in bucket_names]

        x = np.arange(len(bucket_names))
        width = 0.35

        fig, ax = plt.subplots(figsize=(8, 5))
        bars1 = ax.bar(x - width / 2, donor_rates, width, label="Donor",
                       color="#5B9BD5", edgecolor="black", linewidth=0.5)
        bars2 = ax.bar(x + width / 2, acceptor_rates, width, label="Acceptor",
                       color="#ED7D31", edgecolor="black", linewidth=0.5)

        ax.set_ylabel("False Negative Rate")
        ax.set_title(f"Diagnosis 2: FN Rate by {title}")
        ax.set_xticks(x)
        ax.set_xticklabels(bucket_names, rotation=15 if len(bucket_names) > 3 else 0)
        ax.legend()
        ax.set_ylim(0, min(1.0, max(max(donor_rates, default=0),
                                      max(acceptor_rates, default=0)) * 1.3 + 0.05))
        ax.grid(axis="y", alpha=0.3)

        # Add sample counts
        for bar, n in zip(bars1, donor_n):
            if n > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                        f"n={n}", ha="center", va="bottom", fontsize=7)
        for bar, n in zip(bars2, acceptor_n):
            if n > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                        f"n={n}", ha="center", va="bottom", fontsize=7)

        plt.tight_layout()
        plt.savefig(output_dir / filename, dpi=150)
        plt.close()


def plot_diagnosis_3(
    gate_values: np.ndarray,
    all_labels: np.ndarray,
    output_dir: Path,
):
    """Gate distribution histogram and box plot."""
    gate_flat = gate_values.reshape(-1)
    labels_flat = all_labels.reshape(-1)

    gate_splice = gate_flat[labels_flat > 0]
    gate_neither = gate_flat[labels_flat == 0]
    gate_donor = gate_flat[labels_flat == 2]
    gate_acceptor = gate_flat[labels_flat == 1]

    # Histogram overlay (density-normalized)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(gate_neither, bins=100, density=True, alpha=0.5, label="Neither",
            color="#888888")
    ax.hist(gate_splice, bins=100, density=True, alpha=0.7, label="Splice sites",
            color="#E74C3C")
    ax.axvline(gate_neither.mean(), color="#888888", linestyle="--", linewidth=1.5,
               label=f"Neither mean: {gate_neither.mean():.3f}")
    ax.axvline(gate_splice.mean(), color="#E74C3C", linestyle="--", linewidth=1.5,
               label=f"Splice mean: {gate_splice.mean():.3f}")
    ax.set_xlabel("Gate Value")
    ax.set_ylabel("Density")
    ax.set_title("Diagnosis 3: Gate Value Distribution")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "diagnosis3_gate_distribution.png", dpi=150)
    plt.close()

    # Box plot by class
    fig, ax = plt.subplots(figsize=(6, 5))
    # Subsample neither for visualization (too many points)
    rng = np.random.RandomState(42)
    neither_sample = rng.choice(gate_neither, size=min(50000, len(gate_neither)),
                                replace=False)
    data = [neither_sample, gate_acceptor, gate_donor]
    bp = ax.boxplot(data, labels=["Neither\n(sampled)", "Acceptor", "Donor"],
                    patch_artist=True, showfliers=False)
    colors = ["#888888", "#ED7D31", "#5B9BD5"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.set_ylabel("Gate Value")
    ax.set_title("Diagnosis 3: Gate Values by Class")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "diagnosis3_gate_boxplot.png", dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="SpliceMamba Diagnostic Analysis")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best.pt",
                        help="Path to model checkpoint")
    parser.add_argument("--output", type=str, default="diagnostics/",
                        help="Output directory for results and plots")
    args = parser.parse_args()

    cfg = EVAL_CONFIG.copy()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    print("Loading model...")
    model = load_model(args.checkpoint, cfg, device)

    # Run diagnostic inference (single pass)
    print("\nRunning diagnostic inference on test set...")
    coarse_probs, refined_probs = predict_windows_diagnostic(
        model, cfg["test_dataset_path"], cfg, device
    )
    print(f"  Shape: coarse={coarse_probs.shape}, refined={refined_probs.shape}")

    # Read and stitch labels
    print("\nReading labels...")
    all_labels = read_window_labels(cfg["test_dataset_path"])
    windows_per_gene = compute_gene_window_counts(cfg["test_datafile_path"])

    gene_labels = stitch_gene_labels(all_labels, windows_per_gene)
    gene_coarse = stitch_gene_predictions(coarse_probs, windows_per_gene)
    gene_refined = stitch_gene_predictions(refined_probs, windows_per_gene)
    print(f"  Stitched {len(gene_labels)} genes")

    # Run diagnostics
    results = {}
    results["diagnosis_1"] = run_diagnosis_1(gene_coarse, gene_refined, gene_labels)
    results["diagnosis_2"] = run_diagnosis_2(gene_refined, gene_labels, cfg)

    # Generate plots
    print("\nGenerating plots...")
    plot_diagnosis_1(results["diagnosis_1"], output_dir)
    plot_diagnosis_2(results["diagnosis_2"], output_dir)

    # Save results to JSON
    # Convert any non-serializable types
    def make_serializable(obj):
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [make_serializable(v) for v in obj]
        if isinstance(obj, tuple):
            return [make_serializable(v) for v in obj]
        return obj

    results_path = output_dir / "diagnostic_results.json"
    with open(results_path, "w") as f:
        json.dump(make_serializable(results), f, indent=2)
    print(f"\nResults saved to {results_path}")
    print(f"Plots saved to {output_dir}/")
    print("\nDone.")


if __name__ == "__main__":
    main()
