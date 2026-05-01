"""
Convert a (possibly manipulated) datafile_test_*.h5 into the model-ready
dataset_test_*.h5 schema (sharded one-hot 15kb windows + 5kb labels).

For each gene we extract the sense-strand sequence, build a label string of
'0'/'1'/'2' (acceptor/donor) at sense positions, then call OpenSpliceAI's
``create_datapoints`` to produce X (N, 15000, 4) and Y (N, 5000, 3) tensors.

Each gene is written as its own shard (X{i}/Y{i}, i = gene index in datafile),
which preserves stitching order for evaluation/eval_utils.compute_gene_window_counts.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "experiments"))
from manipulate_exons import get_gene_record, gene_to_sense  # noqa: E402

# Inline the OpenSpliceAI encoding functions to avoid pulling gffutils as a hard dep.
# Source: OpenSpliceAI/openspliceai/create_data/utils.py + constants.py
SL = 5000
CL_max = 10000

IN_MAP = np.asarray([
    [0, 0, 0, 0],
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1],
])
OUT_MAP = np.asarray([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
    [0, 0, 0],
])


def _ceil_div(x: int, y: int) -> int:
    return -(-x // y)


def _reformat_data(X0: np.ndarray, Y0: list[np.ndarray]) -> tuple[np.ndarray, list[np.ndarray]]:
    num_points = _ceil_div(len(Y0[0]), SL)
    Xd = np.zeros((num_points, SL + CL_max))
    Yd = [-np.ones((num_points, SL)) for _ in range(1)]
    X0 = np.pad(X0, (0, SL), "constant", constant_values=0)
    Y0 = [np.pad(Y0[t], (0, SL), "constant", constant_values=-1) for t in range(1)]
    for i in range(num_points):
        Xd[i] = X0[SL * i: SL * (i + 1) + CL_max]
        Yd[0][i] = Y0[0][SL * i: SL * (i + 1)]
    return Xd, Yd


def _one_hot_encode(Xd: np.ndarray, Yd: list[np.ndarray]) -> tuple[np.ndarray, list[np.ndarray]]:
    return IN_MAP[Xd.astype("int8")], [OUT_MAP[Yd[t].astype("int8")] for t in range(1)]


def create_datapoints(seq: str, label: str) -> tuple[np.ndarray, list[np.ndarray]]:
    seq = "N" * (CL_max // 2) + seq + "N" * (CL_max // 2)
    seq = seq.upper().replace("A", "1").replace("C", "2").replace("G", "3").replace("T", "4").replace("N", "0")
    label_array = np.array(list(map(int, list(label))))
    X0 = np.asarray(list(map(int, list(seq))))
    Y0 = [label_array]
    Xd, Yd = _reformat_data(X0, Y0)
    return _one_hot_encode(Xd, Yd)


def build_label_string(sense_len: int, donors: list[int], acceptors: list[int]) -> str:
    arr = ["0"] * sense_len
    for a in acceptors:
        if 0 <= a < sense_len:
            arr[a] = "1"
    for d in donors:
        if 0 <= d < sense_len:
            arr[d] = "2"
    return "".join(arr)


def encode_one_gene(rec: dict) -> tuple[np.ndarray, np.ndarray]:
    sense, donors, acceptors = gene_to_sense(rec)
    label = build_label_string(rec["sense_len"], donors, acceptors)
    X, Y = create_datapoints(sense, label)
    # X: (N, 15000, 4), Y: list of one (N, 5000, 3)
    Y_arr = np.asarray(Y[0])  # (N, 5000, 3)
    # Match dataset_test_0.h5 layout: Y is (1, N, 5000, 3) so add leading axis
    Y_arr = Y_arr[np.newaxis, ...]
    return X.astype(np.int8), Y_arr.astype(np.int8)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="in_path", required=True, help="Source datafile_test_*.h5")
    p.add_argument("--out", required=True, help="Output dataset_test_*.h5")
    p.add_argument("--limit", type=int, default=None, help="Process only first N genes")
    args = p.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    src = h5py.File(args.in_path, "r")
    n_genes = src["NAME"].shape[0]
    if args.limit is not None:
        n_genes = min(n_genes, args.limit)

    with h5py.File(out_path, "w") as dst:
        for i in range(n_genes):
            rec = get_gene_record(src, i)
            X, Y = encode_one_gene(rec)
            dst.create_dataset(f"X{i}", data=X, compression=None)
            dst.create_dataset(f"Y{i}", data=Y, compression=None)
            if (i + 1) % 100 == 0 or i + 1 == n_genes:
                print(f"  encoded {i+1}/{n_genes}")
    src.close()
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
