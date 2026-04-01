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

## Analysis

| File | Description |
|---|---|
| `data_analysis.ipynb` | Exploratory data analysis: sequence lengths, intron/exon distributions, nucleotide composition, class balance, k-mer similarity |
| `data_description.md` | Detailed documentation of all data files and their fields |
