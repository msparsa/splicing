"""
CDS-aware manipulations of the SpliceAI test set. Reads `data/cds_masks.json`
(produced by `build_cds_mask.py`) and only operates on genes with a validated
CDS (starts with ATG, length divisible by 3). UTR exonic positions and intronic
positions are left byte-identical to the source.

Modes:
  filtered_baseline  Output the unmodified sequences for every valid-CDS gene
                     (used as the baseline for fair comparison; excludes the
                     ~155 genes that fail validation so all comparisons are
                     on the same gene set).
  nt_shuffle         Permute every nucleotide WITHIN the CDS pool. UTRs and
                     introns unchanged.
  codon_shuffle      Group the CDS sense bases into codons of 3 (in the true
                     reading frame, since the CDS starts with ATG and is
                     mod-3) and permute the codon order. UTRs unchanged.
  remove1            Delete ONE uniformly-random CDS position per gene
                     (per CDS, not per exon). Junctions and TX_END shift.
  remove2            Delete TWO uniformly-random CDS positions per gene.

All four manipulation modes use a fixed seed schedule to be reproducible.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "experiments"))
from manipulate_exons import (  # noqa: E402
    gene_to_sense,
    get_gene_record,
    reverse_complement,
)


DEFAULT_SEEDS = {
    "filtered_baseline": 0,
    "nt_shuffle": 142,
    "codon_shuffle": 143,
    "remove1": 144,
    "remove2": 145,
}


def cds_position_list(cds_intervals: list[list[int]]) -> list[int]:
    """Return sense positions covered by CDS, in ascending order."""
    out: list[int] = []
    for s, e in cds_intervals:
        out.extend(range(s, e))
    return out


def apply_nt_shuffle_cds(sense: str, cds_positions: list[int],
                         rng: np.random.Generator) -> str:
    pool = [sense[p] for p in cds_positions]
    perm = rng.permutation(len(pool))
    new_chars = list(sense)
    for src_idx, dst in zip(perm, cds_positions):
        new_chars[dst] = pool[src_idx]
    return "".join(new_chars)


def apply_codon_shuffle_cds(sense: str, cds_positions: list[int],
                             rng: np.random.Generator) -> str:
    n = len(cds_positions)
    if n % 3 != 0:
        raise ValueError(f"CDS length {n} not divisible by 3 — should have been excluded")
    pool = [sense[p] for p in cds_positions]
    n_codons = n // 3
    codon_order = rng.permutation(n_codons)
    shuffled_pool: list[str] = []
    for ci in codon_order:
        shuffled_pool.extend(pool[ci * 3: (ci + 1) * 3])
    new_chars = list(sense)
    for src_char, dst in zip(shuffled_pool, cds_positions):
        new_chars[dst] = src_char
    return "".join(new_chars)


def apply_remove_k_cds(sense: str, cds_positions: list[int], donors: list[int],
                       acceptors: list[int], k: int,
                       rng: np.random.Generator) -> tuple[str, list[int], list[int], list[dict]]:
    """Delete k uniformly-random positions from the CDS pool. Junctions shift
    via new_pos = old_pos - count(deletions <= old_pos)."""
    n = len(cds_positions)
    n_del = min(k, n)
    if n_del <= 0:
        return sense, donors, acceptors, []
    chosen_idx = rng.choice(n, size=n_del, replace=False)
    delete_set = {cds_positions[i] for i in chosen_idx}
    sorted_dels = sorted(delete_set)

    deletions_meta = []
    for p in sorted_dels:
        rel = cds_positions.index(p)
        deletions_meta.append({
            "abs_pos": int(p),
            "rel_in_cds": int(rel),
            "frac_in_cds": rel / max(n - 1, 1),
        })

    new_chars = [c for i, c in enumerate(sense) if i not in delete_set]
    new_sense = "".join(new_chars)

    def shift(pos: int) -> int:
        lo, hi = 0, len(sorted_dels)
        while lo < hi:
            mid = (lo + hi) // 2
            if sorted_dels[mid] <= pos:
                lo = mid + 1
            else:
                hi = mid
        return pos - lo

    new_donors = [shift(d) for d in donors]
    new_acceptors = [shift(a) for a in acceptors]
    return new_sense, new_donors, new_acceptors, deletions_meta


def sense_to_forward_and_junctions(rec: dict, new_sense: str, new_donors: list[int],
                                   new_acceptors: list[int]) -> tuple[str, list[int], list[int], int]:
    """Same as in manipulate_exons.py — just re-implemented locally to keep this script
    self-contained. sense_len = gene_len + 1 for both strands; new_gene_len = len(new_sense) - 1."""
    new_gene_len = len(new_sense) - 1
    new_tx_end = rec["tx_start"] + new_gene_len
    if rec["strand"] == "+":
        new_gene_fwd = new_sense
        new_jn_start = sorted(rec["tx_start"] + d for d in new_donors)
        new_jn_end = sorted(rec["tx_start"] + a for a in new_acceptors)
    else:
        new_gene_fwd = reverse_complement(new_sense)
        new_jn_start = sorted(new_tx_end - a for a in new_acceptors)
        new_jn_end = sorted(new_tx_end - d for d in new_donors)
    return new_gene_fwd, new_jn_start, new_jn_end, new_tx_end


def manipulate_one(rec: dict, mode: str, cds_intervals: list[list[int]],
                   rng: np.random.Generator) -> dict:
    sense, donors, acceptors = gene_to_sense(rec)
    cds_positions = cds_position_list(cds_intervals)
    deletions_meta: list[dict] = []

    if mode == "filtered_baseline":
        new_sense = sense
        new_donors, new_acceptors = list(donors), list(acceptors)
    elif mode == "nt_shuffle":
        new_sense = apply_nt_shuffle_cds(sense, cds_positions, rng)
        new_donors, new_acceptors = list(donors), list(acceptors)
    elif mode == "codon_shuffle":
        new_sense = apply_codon_shuffle_cds(sense, cds_positions, rng)
        new_donors, new_acceptors = list(donors), list(acceptors)
    elif mode in ("remove1", "remove2"):
        k = 1 if mode == "remove1" else 2
        new_sense, new_donors, new_acceptors, deletions_meta = apply_remove_k_cds(
            sense, cds_positions, donors, acceptors, k, rng)
    else:
        raise ValueError(f"unknown mode {mode!r}")

    new_gene_fwd, new_jn_start, new_jn_end, new_tx_end = sense_to_forward_and_junctions(
        rec, new_sense, new_donors, new_acceptors)

    seq_full = rec["seq_full"]
    left = seq_full[: rec["flank"]]
    right = seq_full[rec["flank"] + rec["sense_len"]:]
    new_seq_full = left + new_gene_fwd + right

    # Sanity checks
    if mode in ("nt_shuffle", "codon_shuffle"):
        cds_orig = [sense[p] for p in cds_positions]
        cds_new = [new_sense[p] for p in cds_positions]
        from collections import Counter
        assert Counter(cds_orig) == Counter(cds_new), "CDS composition changed"
        # UTR + intron must be byte-identical
        cds_set = set(cds_positions)
        for i in range(len(sense)):
            if i not in cds_set:
                assert sense[i] == new_sense[i], f"non-CDS byte changed at {i}"
        assert len(new_sense) == len(sense)
    elif mode in ("remove1", "remove2"):
        k = 1 if mode == "remove1" else 2
        expected = min(k, len(cds_positions))
        assert len(new_sense) == len(sense) - expected, \
            f"unexpected length: {len(new_sense)} vs {len(sense) - expected}"

    return dict(
        name=rec["name"], chrom=rec["chrom"], strand=rec["strand"],
        paralog=rec["paralog"], tx_start=rec["tx_start"],
        new_tx_end=new_tx_end,
        new_seq_full=new_seq_full,
        new_jn_start=new_jn_start,
        new_jn_end=new_jn_end,
        new_gene_len=new_tx_end - rec["tx_start"],
        old_gene_len=rec["gene_len"],
        cds_total_length=len(cds_positions),
        deletions=deletions_meta,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True,
                    choices=["filtered_baseline", "nt_shuffle", "codon_shuffle", "remove1", "remove2"])
    ap.add_argument("--in", dest="in_path", default="datafile_test_0.h5")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cds-masks", default="data/cds_masks.json")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    base_seed = args.seed if args.seed is not None else DEFAULT_SEEDS[args.mode]
    print(f"Mode: {args.mode}  base_seed: {base_seed}")
    print(f"Input:  {args.in_path}")
    print(f"Output: {args.out}")

    cds_masks = json.loads(Path(args.cds_masks).read_text())
    valid_indices = sorted(int(k) for k in cds_masks.keys())
    if args.limit is not None:
        valid_indices = valid_indices[: args.limit]
    print(f"Valid-CDS genes to process: {len(valid_indices):,}")

    src = h5py.File(args.in_path, "r")
    results = []
    new_seq_max = 0
    for j, gi in enumerate(valid_indices):
        rec = get_gene_record(src, gi)
        cds_intervals = cds_masks[str(gi)]["cds_sense_intervals"]
        rng = np.random.default_rng(base_seed + gi)
        out = manipulate_one(rec, args.mode, cds_intervals, rng)
        out["gene_idx"] = gi
        results.append(out)
        if len(out["new_seq_full"]) > new_seq_max:
            new_seq_max = len(out["new_seq_full"])
        if (j + 1) % 200 == 0 or j + 1 == len(valid_indices):
            print(f"  processed {j+1}/{len(valid_indices)}")
    src.close()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    seq_dtype = f"S{new_seq_max}"
    n = len(results)
    with h5py.File(out_path, "w") as dst:
        dst.create_dataset("NAME", data=np.array([r["name"].encode() for r in results], dtype="S16"))
        dst.create_dataset("CHROM", data=np.array([r["chrom"].encode() for r in results], dtype="S4"))
        dst.create_dataset("STRAND", data=np.array([r["strand"].encode() for r in results], dtype="S1"))
        dst.create_dataset("TX_START", data=np.array([r["tx_start"] for r in results], dtype=np.int64))
        dst.create_dataset("TX_END", data=np.array([r["new_tx_end"] for r in results], dtype=np.int64))
        dst.create_dataset("PARALOG", data=np.array([r["paralog"] for r in results], dtype=np.int64))
        dst.create_dataset("SEQ", data=np.array([r["new_seq_full"].encode() for r in results], dtype=seq_dtype))
        dt = h5py.special_dtype(vlen=bytes)
        jn_start_arr = np.empty((n, 1), dtype=object)
        jn_end_arr = np.empty((n, 1), dtype=object)
        for i, r in enumerate(results):
            jn_start_arr[i, 0] = (",".join(str(x) for x in r["new_jn_start"])).encode()
            jn_end_arr[i, 0] = (",".join(str(x) for x in r["new_jn_end"])).encode()
        dst.create_dataset("JN_START", data=jn_start_arr, dtype=dt)
        dst.create_dataset("JN_END", data=jn_end_arr, dtype=dt)
        # Provenance: which row in the new datafile came from which gene_idx of the original
        dst.create_dataset("ORIG_GENE_IDX", data=np.array([r["gene_idx"] for r in results], dtype=np.int64))

    manifest = {
        "mode": args.mode,
        "seed": base_seed,
        "source": str(args.in_path),
        "cds_masks": str(args.cds_masks),
        "n_genes": n,
        "genes": [{
            "gene_idx": r["gene_idx"],
            "name": r["name"],
            "strand": r["strand"],
            "old_gene_len": r["old_gene_len"],
            "new_gene_len": r["new_gene_len"],
            "cds_total_length": r["cds_total_length"],
            "deletions": r["deletions"],
        } for r in results],
    }
    manifest_path = out_path.with_suffix(out_path.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"Wrote datafile: {out_path}")
    print(f"Wrote manifest: {manifest_path}")


if __name__ == "__main__":
    main()
