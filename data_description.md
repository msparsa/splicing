# SpliceAI Data Description

This document describes the four HDF5 data files used for splice site prediction, originally from the SpliceAI project (copied from `/mnt/lareaulab/skang0229/SpliceAI_Py3`).

## Overview

There are two types of files:

- **`datafile_*.h5`** — Raw gene-level data: metadata + full genomic DNA sequences
- **`dataset_*.h5`** — Preprocessed model-ready tensors: one-hot encoded DNA windows + splice site labels

The train/test split follows SpliceAI's convention:
- **Test set:** chromosome 1 (`chr1`)
- **Training set:** all other chromosomes (`chr2`–`chrX`)

---

## `datafile_*.h5` — Raw Gene Data

Each record represents one gene with its genomic context.

| Field | Shape | Dtype | Description |
|---|---|---|---|
| `NAME` | `(N,)` | string | Gene symbol (e.g., `AZIN2`, `NCKAP5`) |
| `CHROM` | `(N,)` | string | Chromosome (e.g., `chr1`, `chr2`) |
| `STRAND` | `(N,)` | string | Strand: `+` or `-` |
| `TX_START` | `(N,)` | int64 | Transcript start coordinate (0-based) |
| `TX_END` | `(N,)` | int64 | Transcript end coordinate |
| `JN_START` | `(N, 1)` | object | Comma-separated intron start positions (string) |
| `JN_END` | `(N, 1)` | object | Comma-separated intron end positions (string) |
| `SEQ` | `(N,)` | string | Raw DNA sequence, padded to fixed width with null bytes |
| `PARALOG` | `(N,)` | int64 | Paralog flag: 0 = no known paralog, 1 = has paralog |

### File-specific details

| File | N (genes) | SEQ max length | Chromosomes |
|---|---|---|---|
| `datafile_test_0.h5` | 1,652 | 1,512,052 bp | chr1 only |
| `datafile_train_all.h5` | 13,384 | 2,102,292 bp | chr2–chrX |

### Notes on `datafile_*`
- The `SEQ` field contains the full genomic region around the transcript, padded to a fixed width. The actual sequence length varies per gene — trailing null bytes (`\x00`) are padding.
- `JN_START` and `JN_END` are stored as comma-separated coordinate strings within an object-typed array. Each pair of corresponding `JN_START[i]` and `JN_END[i]` values defines one intron (donor site at `JN_START`, acceptor site at `JN_END`).
- In the training set, ~70% of genes have `PARALOG=1`. The test set has all `PARALOG=0`. SpliceAI uses paralogs during training to augment data or weight samples.

---

## `dataset_*.h5` — Preprocessed Model Tensors

These files contain sharded, model-ready input/output pairs. Each shard is numbered (`X0/Y0`, `X1/Y1`, ...).

### Input tensors (`X*`)

| Property | Value |
|---|---|
| Shape | `(N_windows, 15000, 4)` |
| Dtype | int8 |
| Values | 0 or 1 (one-hot) |
| Encoding | 4 channels = A, C, G, T |

Each window is a **15,000 bp** stretch of one-hot encoded DNA sequence.

### Output tensors (`Y*`)

| Property | Value |
|---|---|
| Shape | `(1, N_windows, 5000, 3)` |
| Dtype | int8 |
| Values | 0 or 1 (one-hot) |
| Classes | 3 classes (see below) |

Labels are provided only for the **central 5,000 bp** of each 15 kb input window (the flanking 5 kb on each side provide sequence context but are not predicted).

### Label encoding

| One-hot vector | Class | Description |
|---|---|---|
| `[1, 0, 0]` | Neither | Not a splice site |
| `[0, 1, 0]` | Donor | 5' splice site (exon-intron boundary) |
| `[0, 0, 1]` | Acceptor | 3' splice site (intron-exon boundary) |

### File-specific details

| File | Shards | Total windows | Source |
|---|---|---|---|
| `dataset_test_0.h5` | 16 (`X0`–`X15`) | ~16,505 | chr1 genes |
| `dataset_train_all.h5` | 133 (`X0`–`X132`) | ~162,706 | chr2–chrX genes |

### Notes on `dataset_*`
- The leading dimension of `1` in Y tensors is a dummy dimension (likely from how SpliceAI's model outputs predictions).
- Multiple windows can come from a single gene — long transcripts are split into overlapping or adjacent 15 kb windows.
- Windows where the sequence is unknown or at chromosome boundaries are zero-padded (all 4 channels = 0).

---

## Relationship between files

```
datafile_*.h5 (raw genes)
    │
    ├── SEQ: full genomic DNA string per gene
    ├── JN_START / JN_END: intron boundaries → used to create splice site labels
    ├── TX_START / TX_END: define the transcript region
    └── STRAND: determines if sequence needs reverse-complementing
         │
         ▼
    [Preprocessing pipeline]
    - Extract 15 kb windows centered on transcript regions
    - One-hot encode DNA (A/C/G/T → 4 channels)
    - Label central 5 kb positions as donor/acceptor/neither using junction coordinates
         │
         ▼
dataset_*.h5 (model-ready tensors)
    ├── X*: one-hot DNA windows (15000, 4)
    └── Y*: splice site labels (5000, 3)
```

## Using these files for a new model

To train a model with a **different architecture or encoding**:
- Use `datafile_*.h5` as your starting point — it has the raw sequences and junction annotations
- Re-implement the windowing and encoding to suit your model's input format
- Use `JN_START`/`JN_END` to derive splice site labels
- Maintain the same chr1 = test / rest = train split for fair comparison with SpliceAI
