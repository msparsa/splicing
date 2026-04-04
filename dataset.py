"""
SpliceMamba data pipeline.

Loads preprocessed HDF5 windows, applies reverse-complement augmentation,
and provides weighted sampling for class-imbalanced splice site prediction.
"""

from __future__ import annotations


import math
import numpy as np
import h5py
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler


# ---------------------------------------------------------------------------
# Gene-level train / val split utilities
# ---------------------------------------------------------------------------

def compute_gene_window_mapping(datafile_path: str) -> np.ndarray:
    """Return an array of per-gene window counts derived from gene lengths.

    SpliceAI creates ceil(gene_len / 5000) windows per gene.  The cumulative
    sum of these counts lets us map flat window indices back to genes.
    """
    with h5py.File(datafile_path, "r") as f:
        tx_start = f["TX_START"][:]
        tx_end = f["TX_END"][:]
    gene_lens = tx_end - tx_start
    windows_per_gene = np.ceil(gene_lens / 5000).astype(np.int64)
    return windows_per_gene


def gene_level_split(
    datafile_path: str,
    val_fraction: float = 0.1,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Split window indices into train and val sets at the gene level.

    Returns (train_indices, val_indices) — both sorted 1-D int arrays of
    global window indices into the flattened dataset.
    """
    windows_per_gene = compute_gene_window_mapping(datafile_path)
    n_genes = len(windows_per_gene)

    rng = np.random.RandomState(seed)
    perm = rng.permutation(n_genes)
    n_val = max(1, int(n_genes * val_fraction))
    val_genes = set(perm[:n_val].tolist())

    cum = np.cumsum(windows_per_gene)
    starts = np.concatenate([[0], cum[:-1]])

    train_idx, val_idx = [], []
    for g in range(n_genes):
        idxs = np.arange(starts[g], cum[g])
        if g in val_genes:
            val_idx.append(idxs)
        else:
            train_idx.append(idxs)

    train_idx = np.concatenate(train_idx)
    val_idx = np.concatenate(val_idx)
    return np.sort(train_idx), np.sort(val_idx)


# ---------------------------------------------------------------------------
# Sampling weight computation
# ---------------------------------------------------------------------------

def compute_sampling_weights(
    dataset_path: str,
    datafile_path: str,
    indices: np.ndarray | None = None,
) -> np.ndarray:
    """Compute per-window sampling weights.

    * Windows with ≥1 splice site in the label region → weight 5.0
    * Windows with non-canonical splice sites → weight 50.0
    * All other windows → weight 1.0

    Parameters
    ----------
    dataset_path : path to dataset_train_all.h5
    datafile_path : path to datafile_train_all.h5
    indices : optional subset of global window indices to compute weights for.
              If None, computes for all windows.

    Returns
    -------
    weights : 1-D float32 array, one entry per element in *indices* (or all
              windows if indices is None).
    """
    # --- Step 1: build the shard index (same logic as SpliceDataset) ---
    with h5py.File(dataset_path, "r") as f:
        x_keys = sorted(
            [k for k in f.keys() if k.startswith("X")],
            key=lambda k: int(k[1:]),
        )
        shard_sizes = [f[k].shape[0] for k in x_keys]

    total_windows = sum(shard_sizes)
    cum = np.cumsum(shard_sizes)
    starts = np.concatenate([[0], cum[:-1]])

    if indices is None:
        indices = np.arange(total_windows)

    weights = np.ones(len(indices), dtype=np.float32)

    # --- Step 2: identify splice-containing windows via labels ---
    with h5py.File(dataset_path, "r") as f:
        for win_pos, gidx in enumerate(indices):
            # locate shard
            shard_idx = np.searchsorted(cum, gidx, side="right")
            local_idx = int(gidx - starts[shard_idx])
            y_key = f"Y{int(x_keys[shard_idx][1:])}"
            label_onehot = f[y_key][0, local_idx]  # (5000, 3)
            labels = np.argmax(label_onehot, axis=-1)  # (5000,)
            if np.any(labels > 0):
                weights[win_pos] = 5.0

    # --- Step 3: identify non-canonical splice windows ---
    # Build gene→window mapping to know which gene each window belongs to
    windows_per_gene = compute_gene_window_mapping(datafile_path)
    gene_cum = np.cumsum(windows_per_gene)
    gene_starts = np.concatenate([[0], gene_cum[:-1]])

    # Precompute set of non-canonical window indices
    non_canonical_windows = set()
    with h5py.File(datafile_path, "r") as df:
        n_genes = len(windows_per_gene)
        for g in range(n_genes):
            seq_raw = df["SEQ"][g]
            if isinstance(seq_raw, bytes):
                seq = seq_raw.rstrip(b"\x00").decode("ascii", errors="replace")
            else:
                seq = str(seq_raw).rstrip("\x00")

            tx_start = int(df["TX_START"][g])

            jn_start_raw = df["JN_START"][g]
            jn_end_raw = df["JN_END"][g]
            if isinstance(jn_start_raw, np.ndarray):
                jn_start_raw = jn_start_raw[0]
            if isinstance(jn_end_raw, np.ndarray):
                jn_end_raw = jn_end_raw[0]
            if isinstance(jn_start_raw, bytes):
                jn_start_raw = jn_start_raw.decode()
            if isinstance(jn_end_raw, bytes):
                jn_end_raw = jn_end_raw.decode()

            donor_positions = [
                int(x) for x in jn_start_raw.split(",") if x.strip()
            ]
            acceptor_positions = [
                int(x) for x in jn_end_raw.split(",") if x.strip()
            ]

            has_non_canonical = False
            # The sequence is padded by 5000 on each side of the transcript.
            # Position in seq = genomic_coord - tx_start + 5000
            for dp in donor_positions:
                pos_in_seq = dp - tx_start + 5000
                if 0 <= pos_in_seq < len(seq) - 1:
                    dinuc = seq[pos_in_seq : pos_in_seq + 2].upper()
                    if dinuc != "GT":
                        has_non_canonical = True
                        break

            if not has_non_canonical:
                for ap in acceptor_positions:
                    pos_in_seq = ap - tx_start + 5000
                    if 1 <= pos_in_seq <= len(seq):
                        dinuc = seq[pos_in_seq - 2 : pos_in_seq].upper()
                        if dinuc != "AG":
                            has_non_canonical = True
                            break

            if has_non_canonical:
                # All windows of this gene are flagged non-canonical
                for w in range(int(gene_starts[g]), int(gene_cum[g])):
                    non_canonical_windows.add(w)

    # Apply non-canonical weight
    for win_pos, gidx in enumerate(indices):
        if gidx in non_canonical_windows:
            weights[win_pos] = 50.0

    return weights


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class SpliceDataset(Dataset):
    """PyTorch dataset for preprocessed SpliceAI-format HDF5 windows.

    Each sample returns:
        X : (4, 15000) float32 — one-hot encoded DNA
        Y : (5000,)   int64   — class labels {0=neither, 1=donor, 2=acceptor}
    """

    def __init__(
        self,
        dataset_path: str,
        indices: np.ndarray | None = None,
        augment: bool = False,
        rc_prob: float = 0.5,
    ):
        self.dataset_path = dataset_path
        self.augment = augment
        self.rc_prob = rc_prob
        self._file = None  # opened lazily per-worker

        # Build shard index from a temporary handle
        with h5py.File(dataset_path, "r") as f:
            self.x_keys = sorted(
                [k for k in f.keys() if k.startswith("X")],
                key=lambda k: int(k[1:]),
            )
            self.shard_sizes = [f[k].shape[0] for k in self.x_keys]

        self.cum = np.cumsum(self.shard_sizes)
        self.starts = np.concatenate([[0], self.cum[:-1]])
        self.total_windows = int(self.cum[-1])

        if indices is not None:
            self.indices = indices
        else:
            self.indices = np.arange(self.total_windows)

    # --- lazy HDF5 handle (one per worker) ---
    @property
    def file(self):
        if self._file is None:
            self._file = h5py.File(self.dataset_path, "r")
        return self._file

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        gidx = self.indices[idx]

        # Locate shard
        shard_idx = int(np.searchsorted(self.cum, gidx, side="right"))
        local_idx = int(gidx - self.starts[shard_idx])
        shard_num = int(self.x_keys[shard_idx][1:])

        x = self.file[f"X{shard_num}"][local_idx]        # (15000, 4) int8
        y = self.file[f"Y{shard_num}"][0, local_idx]      # (5000, 3) int8

        # Convert to tensors
        x = torch.from_numpy(x.astype(np.float32))        # (15000, 4)
        y = torch.from_numpy(np.argmax(y, axis=-1).astype(np.int64))  # (5000,)

        # Reverse-complement augmentation
        if self.augment and torch.rand(1).item() < self.rc_prob:
            x, y = self._reverse_complement(x, y)

        # Transpose x to channels-first: (4, 15000)
        x = x.permute(1, 0)

        return x, y

    @staticmethod
    def _reverse_complement(x: torch.Tensor, y: torch.Tensor):
        """Apply reverse-complement augmentation.

        x : (15000, 4)  — ACGT one-hot
        y : (5000,)      — class indices {0, 1, 2}
        """
        # 1. Reverse sequence along length
        x = x.flip(0)
        # 2. Swap A↔T (col 0↔3) and C↔G (col 1↔2)
        x = x[:, [3, 2, 1, 0]]
        # 3. Reverse labels
        y = y.flip(0)
        # 4. Swap donor(1) ↔ acceptor(2); neither(0) stays
        swap = y.clone()
        swap[y == 1] = 2
        swap[y == 2] = 1
        y = swap
        return x, y


# ---------------------------------------------------------------------------
# DataLoader builders
# ---------------------------------------------------------------------------

def build_train_loader(
    dataset_path: str,
    datafile_path: str,
    batch_size: int = 8,
    num_workers: int = 4,
    val_fraction: float = 0.1,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader]:
    """Build training and validation DataLoaders with weighted sampling.

    Returns (train_loader, val_loader).
    """
    train_idx, val_idx = gene_level_split(
        datafile_path, val_fraction=val_fraction, seed=seed
    )

    train_ds = SpliceDataset(
        dataset_path, indices=train_idx, augment=True, rc_prob=0.5
    )
    val_ds = SpliceDataset(
        dataset_path, indices=val_idx, augment=False
    )

    # Sampling weights for training set
    print("Computing sampling weights (this may take a few minutes)...")
    weights = compute_sampling_weights(dataset_path, datafile_path, train_idx)
    sampler = WeightedRandomSampler(
        weights=weights.tolist(),
        num_samples=len(weights),
        replacement=True,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
    )

    return train_loader, val_loader


def build_test_loader(
    dataset_path: str,
    batch_size: int = 8,
    num_workers: int = 4,
) -> DataLoader:
    """Build test DataLoader (no augmentation, no sampling weights)."""
    ds = SpliceDataset(dataset_path, augment=False)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
    )
