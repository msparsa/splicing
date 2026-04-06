"""
Compare SpliceAI (5-model ensemble) and SpliceMamba on poison-exon splice sites.

Each poison exon has an acceptor site (3' splice site) and a donor site
(5' splice site).  This script predicts the probability that each model
assigns to those true sites and summarises performance.

Usage:
    # 1) SpliceAI  (run in spliceai_env)
    /mnt/lareaulab/mparsa/miniconda3/envs/spliceai_env/bin/python \
        evaluate_poison_exons.py --model spliceai

    # 2) SpliceMamba  (run in base env)
    python evaluate_poison_exons.py --model splicemamba \
        --checkpoint checkpoints/last.pt

    # 3) Compare  (either env — only needs numpy + matplotlib)
    python evaluate_poison_exons.py --compare
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
from pyfaidx import Fasta

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REF_FASTA = "/mnt/lareaulab/sahilbshah/Pol3_ChromTransfer/data/hg38.fa"
PE_FILE = "reg_pe_cds.txt"
WINDOW = 15000          # total input window for both models
LABEL_START = 5000      # central label region start
LABEL_END = 10000       # central label region end
LABEL_LEN = LABEL_END - LABEL_START  # 5000
BATCH_SIZE = 8


# ---------------------------------------------------------------------------
# Parse poison-exon file
# ---------------------------------------------------------------------------

def parse_poison_exons(path: str) -> list[dict]:
    """Parse reg_pe_cds.txt → list of {chrom, start, end, strand, gene_name}."""
    exons = []
    with open(path) as fh:
        for line in fh:
            if line.startswith("#") or not line.strip():
                continue
            fields = line.strip().split("\t")
            chrom, start, end, strand = fields[0], int(fields[1]), int(fields[2]), fields[4]

            # Extract gene name from annotation field
            gene_name = "unknown"
            for f in fields:
                if "gene_name" in f:
                    # format: "gene_name ""NAME"""
                    parts = f.split('"')
                    for i, p in enumerate(parts):
                        if "gene_name" in p and i + 2 < len(parts):
                            gene_name = parts[i + 2]
                            break
                    if gene_name != "unknown":
                        break

            exons.append(dict(
                chrom=chrom, start=start, end=end,
                strand=strand, gene_name=gene_name,
            ))
    return exons


# ---------------------------------------------------------------------------
# Sequence extraction & one-hot encoding
# ---------------------------------------------------------------------------

_ENCODE = np.zeros((256, 4), dtype=np.float32)
_ENCODE[ord("A")] = [1, 0, 0, 0]
_ENCODE[ord("a")] = [1, 0, 0, 0]
_ENCODE[ord("C")] = [0, 1, 0, 0]
_ENCODE[ord("c")] = [0, 1, 0, 0]
_ENCODE[ord("G")] = [0, 0, 1, 0]
_ENCODE[ord("g")] = [0, 0, 1, 0]
_ENCODE[ord("T")] = [0, 0, 0, 1]
_ENCODE[ord("t")] = [0, 0, 0, 1]


def one_hot_encode(seq: str) -> np.ndarray:
    """Encode a DNA string as (L, 4) float32.  Non-ACGT bases → all zeros."""
    return _ENCODE[np.frombuffer(seq.encode("ascii"), dtype=np.uint8)]


def prepare_windows(exons: list[dict], ref_fasta: Fasta):
    """For each exon, extract a 15 kb window and record splice-site positions.

    Returns
    -------
    windows : np.ndarray, (N, 15000, 4)
    acc_pos : np.ndarray, (N,)  — position of acceptor within label region
    don_pos : np.ndarray, (N,)  — position of donor within label region
    kept_idx : list[int]        — indices into original exon list that were kept
    """
    windows, acc_pos, don_pos, kept_idx = [], [], [], []
    half = WINDOW // 2  # 7500

    for i, ex in enumerate(exons):
        chrom = ex["chrom"]
        exon_mid = (ex["start"] + ex["end"]) // 2
        win_start = exon_mid - half
        win_end = exon_mid + half

        # Splice-site genomic positions (0-based).
        # Coordinates in the file are 1-based GTF.
        # In SpliceAI's convention:
        #   acceptor label = first exon base (0-based: start - 1)
        #   donor label    = last exon base  (0-based: end - 1)
        if ex["strand"] == "+":
            acc_genomic = ex["start"] - 1   # first exon base (0-based)
            don_genomic = ex["end"] - 1     # last exon base (0-based)
        else:
            don_genomic = ex["start"] - 1   # last exon base on - strand
            acc_genomic = ex["end"] - 1     # first exon base on - strand

        # Positions within the 15 kb window
        acc_win = acc_genomic - win_start
        don_win = don_genomic - win_start

        # Both sites must be in the label region [5000, 10000)
        acc_label = acc_win - LABEL_START
        don_label = don_win - LABEL_START
        if not (0 <= acc_label < LABEL_LEN and 0 <= don_label < LABEL_LEN):
            continue

        # Check genome bounds
        if win_start < 0:
            continue
        try:
            chrom_len = len(ref_fasta[chrom])
        except KeyError:
            continue
        if win_end > chrom_len:
            continue

        seq = str(ref_fasta[chrom][win_start:win_end])
        if len(seq) != WINDOW:
            continue

        encoded = one_hot_encode(seq)
        windows.append(encoded)
        acc_pos.append(acc_label)
        don_pos.append(don_label)
        kept_idx.append(i)

    windows = np.stack(windows, axis=0)  # (N, 15000, 4)
    return windows, np.array(acc_pos), np.array(don_pos), kept_idx


# ---------------------------------------------------------------------------
# SpliceAI inference
# ---------------------------------------------------------------------------

def run_spliceai(windows: np.ndarray) -> np.ndarray:
    """Run 5-model SpliceAI ensemble.  Returns (N, 5000, 3) softmax probs."""
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    import tensorflow as tf
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    from keras.models import load_model
    from pkg_resources import resource_filename

    models = []
    for i in range(1, 6):
        path = resource_filename("spliceai", f"models/spliceai{i}.h5")
        print(f"  Loading {path}")
        models.append(load_model(path, compile=False))

    n = windows.shape[0]
    ensemble_probs = np.zeros((n, LABEL_LEN, 3), dtype=np.float32)

    for model in models:
        preds = []
        for start in range(0, n, BATCH_SIZE):
            batch = windows[start : start + BATCH_SIZE]
            preds.append(model.predict(batch, verbose=0))
        ensemble_probs += np.concatenate(preds, axis=0)

    ensemble_probs /= len(models)
    return ensemble_probs  # (N, 5000, 3)


# ---------------------------------------------------------------------------
# SpliceMamba inference
# ---------------------------------------------------------------------------

def run_splicemamba(windows: np.ndarray, checkpoint: str) -> np.ndarray:
    """Run SpliceMamba.  Returns (N, 5000, 3) softmax probs."""
    import torch
    from model import SpliceMamba

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SpliceMamba(
        d_model=256, n_mamba_layers=8, d_state=64, expand=2,
        d_conv=4, headdim=32, n_attn_layers=4, n_heads=8,
        window_radius=400, dropout=0.1, n_classes=3, max_len=15000,
    ).to(device)

    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"  Loaded checkpoint (epoch {ckpt.get('epoch', '?')})")

    n = windows.shape[0]
    all_probs = []

    with torch.no_grad():
        for start in range(0, n, BATCH_SIZE):
            # windows is (N, 15000, 4) — transpose to (B, 4, 15000)
            batch = torch.from_numpy(
                windows[start : start + BATCH_SIZE]
            ).permute(0, 2, 1).to(device)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                _, refined_logits, _ = model(batch)

            label_logits = refined_logits[:, LABEL_START:LABEL_END, :]
            probs = torch.softmax(label_logits.float(), dim=-1)
            all_probs.append(probs.cpu().numpy())

    return np.concatenate(all_probs, axis=0)  # (N, 5000, 3)


# ---------------------------------------------------------------------------
# Save / load predictions
# ---------------------------------------------------------------------------

def save_preds(model_name: str, exons: list[dict], kept_idx: list[int],
               acc_pos: np.ndarray, don_pos: np.ndarray,
               probs: np.ndarray):
    out = f"poison_exon_preds_{model_name}.npz"
    kept_exons = [exons[i] for i in kept_idx]
    np.savez_compressed(
        out,
        probs=probs,           # (N, 5000, 3)
        acc_pos=acc_pos,       # (N,)
        don_pos=don_pos,       # (N,)
        chroms=np.array([e["chrom"] for e in kept_exons]),
        starts=np.array([e["start"] for e in kept_exons]),
        ends=np.array([e["end"] for e in kept_exons]),
        strands=np.array([e["strand"] for e in kept_exons]),
        gene_names=np.array([e["gene_name"] for e in kept_exons]),
    )
    print(f"Saved {probs.shape[0]} predictions → {out}")


def load_preds(model_name: str) -> dict:
    path = f"poison_exon_preds_{model_name}.npz"
    data = np.load(path, allow_pickle=True)
    return dict(data)


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

def compare():
    spliceai = load_preds("spliceai")
    splicemamba = load_preds("splicemamba")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    results = {}

    for name, data in [("SpliceAI", spliceai), ("SpliceMamba", splicemamba)]:
        probs = data["probs"]          # (N, 5000, 3)
        acc_pos = data["acc_pos"]      # (N,)
        don_pos = data["don_pos"]      # (N,)
        n = probs.shape[0]

        # Extract predicted probabilities at true splice sites
        acc_probs = probs[np.arange(n), acc_pos, 1]  # class 1 = acceptor
        don_probs = probs[np.arange(n), don_pos, 2]  # class 2 = donor

        results[name] = dict(acc_probs=acc_probs, don_probs=don_probs)

    # Print comparison table
    print("=" * 70)
    print("POISON EXON SPLICE SITE COMPARISON")
    print(f"  Exons evaluated: {spliceai['probs'].shape[0]}")
    print("=" * 70)

    for site, cls_name in [("acc", "Acceptor"), ("don", "Donor")]:
        key = f"{site}_probs"
        print(f"\n--- {cls_name} sites ---")
        print(f"{'Metric':<35} {'SpliceAI':>12} {'SpliceMamba':>12}")
        print("-" * 60)

        sa = results["SpliceAI"][key]
        sm = results["SpliceMamba"][key]

        print(f"{'Mean probability':<35} {sa.mean():>12.4f} {sm.mean():>12.4f}")
        print(f"{'Median probability':<35} {np.median(sa):>12.4f} {np.median(sm):>12.4f}")
        print(f"{'Std dev':<35} {sa.std():>12.4f} {sm.std():>12.4f}")

        for thresh in [0.1, 0.3, 0.5, 0.7]:
            sa_rate = (sa >= thresh).mean()
            sm_rate = (sm >= thresh).mean()
            print(f"{'Detection rate (≥' + str(thresh) + ')':<35} "
                  f"{sa_rate:>11.1%} {sm_rate:>12.1%}")

    # --- Figures ---
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    for col, (site, cls_name) in enumerate([("acc", "Acceptor"), ("don", "Donor")]):
        key = f"{site}_probs"
        sa = results["SpliceAI"][key]
        sm = results["SpliceMamba"][key]

        # Row 0: histograms
        ax = axes[0, col]
        bins = np.linspace(0, 1, 51)
        ax.hist(sa, bins=bins, alpha=0.6, label="SpliceAI", color="tab:blue")
        ax.hist(sm, bins=bins, alpha=0.6, label="SpliceMamba", color="tab:orange")
        ax.set_xlabel("Predicted probability")
        ax.set_ylabel("Count")
        ax.set_title(f"{cls_name} site probabilities")
        ax.legend()

        # Row 1: scatter (SpliceAI vs SpliceMamba)
        ax = axes[1, col]
        ax.scatter(sa, sm, alpha=0.15, s=8, color="tab:purple")
        ax.plot([0, 1], [0, 1], "k--", lw=0.8)
        ax.set_xlabel("SpliceAI probability")
        ax.set_ylabel("SpliceMamba probability")
        ax.set_title(f"{cls_name}: SpliceAI vs SpliceMamba")
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_aspect("equal")

    plt.tight_layout()
    out_fig = "poison_exon_comparison.png"
    plt.savefig(out_fig, dpi=150)
    print(f"\nFigure saved → {out_fig}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Compare SpliceAI & SpliceMamba on poison-exon splice sites")
    parser.add_argument("--model", choices=["spliceai", "splicemamba"],
                        help="Run inference with this model and save predictions")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/last.pt",
                        help="SpliceMamba checkpoint path (default: checkpoints/last.pt)")
    parser.add_argument("--compare", action="store_true",
                        help="Load saved predictions from both models and compare")
    args = parser.parse_args()

    if not args.model and not args.compare:
        parser.error("Specify --model spliceai, --model splicemamba, or --compare")

    if args.compare:
        compare()
        return

    # --- Shared data preparation ---
    print("Parsing poison exon file...")
    exons = parse_poison_exons(PE_FILE)
    print(f"  {len(exons)} exons in file")

    print("Loading reference genome...")
    ref = Fasta(REF_FASTA)

    print("Preparing 15 kb windows...")
    windows, acc_pos, don_pos, kept_idx = prepare_windows(exons, ref)
    print(f"  {windows.shape[0]} exons kept (both sites in central 5 kb)")

    # --- Model-specific inference ---
    if args.model == "spliceai":
        print("\nRunning SpliceAI 5-model ensemble...")
        probs = run_spliceai(windows)
    else:
        print("\nRunning SpliceMamba...")
        probs = run_splicemamba(windows, args.checkpoint)

    save_preds(args.model, exons, kept_idx, acc_pos, don_pos, probs)

    # Quick summary
    n = probs.shape[0]
    acc_probs = probs[np.arange(n), acc_pos, 1]
    don_probs = probs[np.arange(n), don_pos, 2]
    print(f"\nQuick summary ({args.model}):")
    print(f"  Acceptor — mean: {acc_probs.mean():.4f}, "
          f"detected (≥0.5): {(acc_probs >= 0.5).mean():.1%}")
    print(f"  Donor    — mean: {don_probs.mean():.4f}, "
          f"detected (≥0.5): {(don_probs >= 0.5).mean():.1%}")


if __name__ == "__main__":
    main()
