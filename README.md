# Splice Site Prediction

Predicting splice sites (donor and acceptor) from pre-mRNA sequences using deep learning. The goal is the same task as [SpliceAI](https://doi.org/10.1016/j.cell.2018.12.015) — given a raw DNA sequence, classify each position as a **donor (5') splice site**, **acceptor (3') splice site**, or **neither** — but with a different model architecture.

We use the same training and test data as SpliceAI to enable direct comparison.

## Data

The data originates from SpliceAI's preprocessing pipeline (source: `/mnt/lareaulab/skang0229/SpliceAI_Py3`). The train/test split follows the SpliceAI convention: **chr1 = test**, all other chromosomes = train.

| File | Description |
|---|---|
| `datafile_train_all.h5` | Raw gene data for training (13,384 genes): sequences, transcript coordinates, intron junctions, metadata |
| `datafile_test_0.h5` | Raw gene data for testing (1,652 genes on chr1) |
| `dataset_train_all.h5` | SpliceAI's preprocessed training tensors: one-hot encoded 15kb DNA windows + 3-class splice site labels |
| `dataset_test_0.h5` | SpliceAI's preprocessed test tensors |

See [data_description.md](data_description.md) for detailed schema documentation.

## Model Architecture

### Version 1 (v1): Gated SpliceMamba — 3.5M params

```
Input (B, 4, 15000)
  → Conv Stem (4→128→256, kernel 11)
  → Sinusoidal Positional Encoding
  → BiMamba1 Encoder (6 layers/direction, D=256, N=16, E=1)
  → Coarse Head (MLP → 3 classes)
  → Gate = sigmoid(max(donor, acceptor logit) / τ)
  → Gated Sliding-Window Attention (2 layers, R=200, 8 heads)
  → Refined Head (MLP → 3 classes)
```

- **Encoder**: Bidirectional Mamba (v1) with 6 layers per direction, d_state=16, expand=1. Forward and backward stacks fused via linear projection.
- **Attention**: Gated sliding-window self-attention using FlashAttention-2. A sigmoid gate derived from coarse logits controlled how much attention output was mixed with the encoder representation.
- **Training**: Focal loss (gamma=2.0, alpha=[0.01, 1.0, 1.0]), AdamW (lr=3e-4), warmup+cosine schedule, gate temperature annealed from τ=5.0→1.0 over 10 epochs.
- **Best result**: Validation AUPRC mean = **0.9303** at epoch 32 (~39 hours total training on A100 80GB).

### Version 2 (v2): SpliceMamba2 — 8.3M params (current)

```
Input (B, 4, 15000)
  → Multi-scale Conv Stem (parallel kernel 5 + kernel 11, → 256)
  → Sinusoidal Positional Encoding
  → BiMamba2 Encoder (8 layers/direction, D=256, N=64, E=2, headdim=32)
  → Coarse Head (auxiliary loss only, decoupled)
  → Sliding-Window Attention (4 layers, R=400, 8 heads, standard residual)
  → Refined Head (MLP → 3 classes)
```

- **Encoder**: Bidirectional Mamba2 (SSD) with 8 layers per direction, d_state=64, expand=2, headdim=32. ~2x faster training throughput vs Mamba1.
- **Attention**: Standard pre-norm residual sliding-window attention (no gating). 4 layers with window radius 400bp.
- **Conv Stem**: Multi-scale with parallel kernel_size=5 and kernel_size=11 branches for better local motif capture around splice sites.
- **Training**: Same loss and optimizer as v1, no gate temperature annealing.

### Why v2? Diagnostic Findings

We ran three diagnostic analyses on v1's test set predictions (`diagnose.py`) that revealed:

**1. The gate mechanism was broken (d' = -0.14)**

The gate was supposed to focus attention on predicted splice sites, but it was actually *anti-correlated* — gate values at splice sites (mean 0.285) were **lower** than at non-splice positions (mean 0.298). Only 0.014% of splice-site gates exceeded 0.5, and 0% exceeded 0.7. The coarse head couldn't produce confident enough logits to drive a useful gate. Attention contributed only +2.4% AUPRC over the coarse head alone — likely from residual connection capacity, not the gating mechanism.

**2. Errors concentrated on short introns and tiny exons**

| Category | FN Rate |
|----------|---------|
| Short introns (<200bp) | 6.3–8.2% |
| Medium introns (1k–5k bp) | 3.4–4.1% |
| Tiny exons (<80bp) | 7.2–8.3% |
| Normal exons (80–200bp) | 3.2–4.3% |

This is a **local pattern recognition** problem, not a long-range dependency problem. The Mamba encoder handled long introns well. The failures occurred where donor and acceptor signals overlap within small windows.

**3. No edge effects — window size is adequate**

Error rates were flat across the 5kb label region (4.0–4.9%), with only a slight elevation at the 0–20% position for acceptors (7.5%). The 15kb window with 5kb flanks provides sufficient context.

**v2 changes based on these findings:**
- Removed the broken gate mechanism entirely; replaced with standard pre-norm residual attention
- Added multi-scale conv stem (kernel 5 + kernel 11) to improve local motif detection for the short-intron/tiny-exon weakness
- Upgraded to Mamba2 for faster training
- Increased attention depth (2→4 layers) and window (R=200→400) to give attention a real chance
- Increased encoder depth (6→8 layers/direction) for more representational capacity

## Analysis

| File | Description |
|---|---|
| `data_analysis.ipynb` | Exploratory data analysis: sequence lengths, intron/exon distributions, nucleotide composition, class balance, k-mer similarity |
| `data_description.md` | Detailed documentation of all data files and their fields |
| `diagnose.py` | Diagnostic analysis of v1 model: attention contribution, error categorization by structural features |
