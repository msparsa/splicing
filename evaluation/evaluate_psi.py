"""
PSI correlation evaluation — compare SpliceMamba against TrASPr, Pangolin,
and SpliceAI on cassette exon PSI prediction.

Uses TrASPr's pre-existing GTEx evaluation data (GTEx_data.tsv) which already
has PSI labels and predictions from TrASPr, Pangolin, SpliceAI, and SpliceTF.
This script adds SpliceMamba predictions and computes correlation metrics.

For each cassette exon event:
  - Extract 15000bp sequence centered on exon start (acceptor) and exon end (donor)
  - Run SpliceMamba inference to get P(acceptor) at exon start, P(donor) at exon end
  - Score = average of the two splice site probabilities
  - Correlate with PSI labels

Usage:
    python evaluate_psi.py --checkpoint checkpoints/best.pt
    python evaluate_psi.py --checkpoint checkpoints/best.pt --ref-genome /path/to/hg38.fa
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np
import pysam
import torch
from scipy.stats import pearsonr, spearmanr

from model import SpliceMamba
from eval_utils import make_serializable

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

GTEX_DATA_PATH = "TrASPr/plot_script/data/GTEx_data.tsv"
REF_GENOME_PATH = "hg38.fa"  # symlink to /mnt/lareaulab/carmelle/genomes/hg38.fa

TISSUE_MAP = {
    "Heart_Atrial_Appendage": "heart",
    "Brain_Cerebellum": "brain",
    "Liver": "liver",
}

MODEL_CONFIG = dict(
    d_model=256,
    n_mamba_layers=8,
    d_state=64,
    expand=2,
    d_conv=4,
    headdim=32,
    n_attn_layers=4,
    n_heads=8,
    window_radius=400,
    dropout=0.15,
    drop_path_rate=0.0,
    n_classes=3,
    max_len=15000,
    label_start=5000,
    label_end=10000,
)

# One-hot encoding for DNA
BASE_MAP = {"A": 0, "C": 1, "G": 2, "T": 3, "N": -1}


# ---------------------------------------------------------------------------
# Load GTEx PSI data
# ---------------------------------------------------------------------------

def load_gtex_data(data_path: str) -> list[dict]:
    """Load TrASPr's GTEx cassette exon data with existing model predictions.

    Each event has: ID, Chr, Strand, Exon_start, Exon_end, Tissue,
    Change_case, Label (PSI), TrASPr_pred, Pangolin_pred, SpliceAI_pred,
    SpliceTF_pred.
    """
    events = []
    with open(data_path) as f:
        header = f.readline().strip().split("\t")
        for line in f:
            fields = line.strip().split("\t")
            row = dict(zip(header, fields))
            row["Label"] = float(row["Label"])
            row["Exon_start"] = int(row["Exon_start"])
            row["Exon_end"] = int(row["Exon_end"])
            # Parse existing predictions
            for key in ["TrASPr_pred", "Pangolin_pred", "SpliceAI_pred", "SpliceTF_pred"]:
                if key in row:
                    try:
                        row[key] = float(row[key])
                    except (ValueError, KeyError):
                        row[key] = float("nan")
            events.append(row)

    print(f"Loaded {len(events)} cassette exon events")
    for tissue in sorted(set(e["Tissue"] for e in events)):
        n = sum(1 for e in events if e["Tissue"] == tissue)
        print(f"  {tissue}: {n} events")
    return events


# ---------------------------------------------------------------------------
# Sequence extraction
# ---------------------------------------------------------------------------

def one_hot_encode(seq: str) -> np.ndarray:
    """Encode DNA sequence as one-hot (L, 4) array."""
    arr = np.zeros((len(seq), 4), dtype=np.float32)
    for i, base in enumerate(seq.upper()):
        idx = BASE_MAP.get(base, -1)
        if idx >= 0:
            arr[i, idx] = 1.0
    return arr


def extract_sequence(
    fasta: pysam.FastaFile,
    chrom: str,
    center: int,
    window: int = 15000,
) -> str:
    """Extract a sequence centered on a genomic position.

    Pads with N if the window extends beyond chromosome boundaries.
    """
    half = window // 2
    start = center - half
    end = center + half

    chrom_len = fasta.get_reference_length(chrom)

    # Handle boundaries
    pad_left = max(0, -start)
    pad_right = max(0, end - chrom_len)
    fetch_start = max(0, start)
    fetch_end = min(chrom_len, end)

    seq = fasta.fetch(chrom, fetch_start, fetch_end)
    seq = "N" * pad_left + seq + "N" * pad_right

    return seq[:window]


# ---------------------------------------------------------------------------
# SpliceMamba inference on cassette exon events
# ---------------------------------------------------------------------------

@torch.no_grad()
def score_cassette_exons(
    model: torch.nn.Module,
    events: list[dict],
    fasta: pysam.FastaFile,
    device: torch.device,
    batch_size: int = 16,
) -> list[float]:
    """Score each cassette exon event with SpliceMamba.

    For each event:
    1. Extract 15000bp sequence centered on exon start and exon end
    2. Always use forward-strand sequence (model handles both orientations)
    3. P(splice) = P(acceptor) + P(donor) at center position
    4. Score = average of P(splice) at exon_start and exon_end
    """
    scores = []
    n_events = len(events)

    # Deduplicate: group events by unique (Chr, Strand, Exon_start, Exon_end)
    unique_exons = {}
    for i, event in enumerate(events):
        key = (event["Chr"], event["Strand"],
               event["Exon_start"], event["Exon_end"])
        if key not in unique_exons:
            unique_exons[key] = []
        unique_exons[key].append(i)

    print(f"  {len(unique_exons)} unique exons from {n_events} events")

    exon_scores = {}
    exon_keys = list(unique_exons.keys())

    for batch_start in range(0, len(exon_keys), batch_size):
        batch_end = min(batch_start + batch_size, len(exon_keys))
        batch_keys = exon_keys[batch_start:batch_end]

        # Prepare sequences centered on exon_start and exon_end
        # Use forward strand — the model handles both strand orientations
        start_seqs = []
        end_seqs = []
        for chrom, strand, exon_start, exon_end in batch_keys:
            seq_start = extract_sequence(fasta, chrom, exon_start)
            seq_end = extract_sequence(fasta, chrom, exon_end)
            start_seqs.append(one_hot_encode(seq_start))
            end_seqs.append(one_hot_encode(seq_end))

        # Run model on exon-start-centered sequences
        start_batch = torch.from_numpy(
            np.stack(start_seqs)
        ).permute(0, 2, 1).to(device)  # (B, 4, 15000)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            _, refined_start, _ = model(start_batch)

        # Run model on exon-end-centered sequences
        end_batch = torch.from_numpy(
            np.stack(end_seqs)
        ).permute(0, 2, 1).to(device)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            _, refined_end, _ = model(end_batch)

        # Center of 15000bp input = position 7500
        center_idx = 7500

        # Softmax over full sequence
        start_probs = torch.softmax(refined_start.float(), dim=-1)
        end_probs = torch.softmax(refined_end.float(), dim=-1)

        for b_idx, key in enumerate(batch_keys):
            # P(splice) = P(acceptor) + P(donor) at center position
            p_splice_start = (start_probs[b_idx, center_idx, 1].item() +
                              start_probs[b_idx, center_idx, 2].item())
            p_splice_end = (end_probs[b_idx, center_idx, 1].item() +
                            end_probs[b_idx, center_idx, 2].item())

            score = (p_splice_start + p_splice_end) / 2.0
            exon_scores[key] = score

        if (batch_start // batch_size) % 10 == 0:
            print(f"    Processed {batch_end}/{len(exon_keys)} unique exons")

    # Map back to all events
    for event in events:
        key = (event["Chr"], event["Strand"],
               event["Exon_start"], event["Exon_end"])
        scores.append(exon_scores[key])

    return scores


# ---------------------------------------------------------------------------
# Correlation metrics
# ---------------------------------------------------------------------------

def compute_correlations(
    labels: np.ndarray,
    predictions: np.ndarray,
    model_name: str,
) -> dict:
    """Compute Pearson and Spearman correlations."""
    mask = ~(np.isnan(labels) | np.isnan(predictions))
    labels = labels[mask]
    predictions = predictions[mask]

    if len(labels) < 3:
        return {"n": 0, "pearson_r": 0.0, "spearman_rho": 0.0, "mse": 0.0}

    r, p_val_r = pearsonr(labels, predictions)
    rho, p_val_rho = spearmanr(labels, predictions)
    mse = float(np.mean((labels - predictions) ** 2))

    return {
        "n": int(len(labels)),
        "pearson_r": float(r),
        "pearson_p": float(p_val_r),
        "spearman_rho": float(rho),
        "spearman_p": float(p_val_rho),
        "mse": mse,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def evaluate_psi(
    checkpoint_path: str,
    gtex_data_path: str = GTEX_DATA_PATH,
    ref_genome_path: str = REF_GENOME_PATH,
    output_dir: str = "results",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    start_time = time.time()

    # Load GTEx cassette exon data
    print("=" * 60)
    print("PSI CORRELATION EVALUATION")
    print("=" * 60)
    events = load_gtex_data(gtex_data_path)

    # Load SpliceMamba model
    print(f"\nLoading SpliceMamba from {checkpoint_path}...")
    from evaluate import load_model
    model = load_model(checkpoint_path, MODEL_CONFIG, device)

    # Open reference genome
    print(f"Opening reference genome: {ref_genome_path}")
    fasta = pysam.FastaFile(ref_genome_path)

    # Score all cassette exons with SpliceMamba
    print("\nScoring cassette exons with SpliceMamba...")
    sm_scores = score_cassette_exons(model, events, fasta, device)

    del model
    torch.cuda.empty_cache()
    fasta.close()

    # Add SpliceMamba scores to events
    for i, score in enumerate(sm_scores):
        events[i]["SpliceMamba_pred"] = score

    # Compute correlations per tissue and overall
    model_keys = [
        ("SpliceMamba_pred", "SpliceMamba"),
        ("TrASPr_pred", "TrASPr"),
        ("Pangolin_pred", "Pangolin"),
        ("SpliceAI_pred", "SpliceAI"),
        ("SpliceTF_pred", "SpliceTF"),
    ]

    results = {"overall": {}, "per_tissue": {}}

    # Overall correlations
    print(f"\n{'='*60}")
    print("OVERALL PSI CORRELATIONS")
    print(f"{'='*60}")
    print(f"{'Model':<15} {'N':>6} {'Pearson r':>12} {'Spearman rho':>14} {'MSE':>10}")
    print("-" * 60)

    labels_all = np.array([e["Label"] for e in events])

    for pred_key, model_name in model_keys:
        preds = np.array([e.get(pred_key, float("nan")) for e in events])
        corr = compute_correlations(labels_all, preds, model_name)
        results["overall"][model_name] = corr
        print(f"{model_name:<15} {corr['n']:>6} {corr['pearson_r']:>12.4f} "
              f"{corr['spearman_rho']:>14.4f} {corr['mse']:>10.6f}")

    # Per-tissue correlations
    tissues = sorted(set(e["Tissue"] for e in events))
    for tissue in tissues:
        tissue_key = TISSUE_MAP.get(tissue, tissue.lower())
        tissue_events = [e for e in events if e["Tissue"] == tissue]
        labels_t = np.array([e["Label"] for e in tissue_events])

        print(f"\n{'='*60}")
        print(f"PSI CORRELATIONS — {tissue} ({len(tissue_events)} events)")
        print(f"{'='*60}")
        print(f"{'Model':<15} {'Pearson r':>12} {'Spearman rho':>14} {'MSE':>10}")
        print("-" * 55)

        results["per_tissue"][tissue] = {}
        for pred_key, model_name in model_keys:
            preds = np.array([e.get(pred_key, float("nan")) for e in tissue_events])
            corr = compute_correlations(labels_t, preds, model_name)
            results["per_tissue"][tissue][model_name] = corr
            print(f"{model_name:<15} {corr['pearson_r']:>12.4f} "
                  f"{corr['spearman_rho']:>14.4f} {corr['mse']:>10.6f}")

    elapsed = time.time() - start_time
    print(f"\nTotal time: {elapsed:.1f}s")

    # Save results
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    all_results = {
        "model": "psi_correlation",
        "timestamp": datetime.now().isoformat(),
        "n_events": len(events),
        "gtex_data_path": gtex_data_path,
        "results": results,
    }

    json_path = out_path / "psi_correlation_results.json"
    with open(json_path, "w") as fp:
        json.dump(make_serializable(all_results), fp, indent=2)
    print(f"\nResults saved to {json_path}")

    # Also save the augmented events TSV with SpliceMamba predictions
    augmented_path = out_path / "GTEx_data_with_splicemamba.tsv"
    with open(augmented_path, "w") as f:
        headers = ["ID", "Chr", "Strand", "Exon_start", "Exon_end", "Tissue",
                    "Change_case", "Label", "TrASPr_pred", "Pangolin_pred",
                    "SpliceAI_pred", "SpliceTF_pred", "SpliceMamba_pred"]
        f.write("\t".join(headers) + "\n")
        for e in events:
            row = [str(e.get(h, "")) for h in headers]
            f.write("\t".join(row) + "\n")
    print(f"Augmented data saved to {augmented_path}")

    return all_results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="PSI correlation evaluation (TrASPr comparison)"
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="SpliceMamba checkpoint path",
    )
    parser.add_argument(
        "--gtex-data", type=str, default=GTEX_DATA_PATH,
        help=f"Path to GTEx_data.tsv (default: {GTEX_DATA_PATH})",
    )
    parser.add_argument(
        "--ref-genome", type=str, default=REF_GENOME_PATH,
        help=f"Path to hg38 reference FASTA (default: {REF_GENOME_PATH})",
    )
    parser.add_argument(
        "--output-dir", type=str, default="results",
        help="Output directory (default: results/)",
    )
    args = parser.parse_args()

    evaluate_psi(
        checkpoint_path=args.checkpoint,
        gtex_data_path=args.gtex_data,
        ref_genome_path=args.ref_genome,
        output_dir=args.output_dir,
    )
