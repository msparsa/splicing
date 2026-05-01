"""
Apply one of four sequence-level manipulations to the exonic content of every
gene in datafile_test_0.h5 and write a new datafile in the same schema.

Modes:
  nt_shuffle    Concatenate all exons (sense strand), randomly permute every
                nucleotide, write back. Junction positions unchanged.
  codon_shuffle Concatenate all exons, group into codons (frame 0 from start
                of concatenated exonic sequence), permute codon order. Trailing
                1-2 nt remainder is left in place. Junctions unchanged.
  remove1       Delete one uniformly-random nucleotide per exon. Junction
                positions and TX_END are shifted to remain consistent with the
                shorter sequence.
  remove2       Same as remove1 but two nt per exon.

Sequence convention (verified empirically against dataset_test_0.h5):
  - SEQ in datafile is forward genomic strand with 5000 bp real flank on each
    side (padded with null bytes to a fixed width).
  - For + strand: sense = forward; sense_pos = genomic - TX_START.
    JN_START = donor (last exon nt). JN_END = acceptor (first nt of new exon).
  - For - strand: sense = reverse-complement of forward.
    sense_pos = TX_END - genomic. JN_START = acceptor (sense). JN_END = donor (sense).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np

DEFAULT_SEEDS = {
    "nt_shuffle": 42,
    "codon_shuffle": 43,
    "remove1": 44,
    "remove2": 45,
}

_COMPLEMENT = str.maketrans("ACGTNacgtn", "TGCANtgcan")


def reverse_complement(seq: str) -> str:
    return seq.translate(_COMPLEMENT)[::-1]


def parse_int_csv(raw) -> list[int]:
    if isinstance(raw, np.ndarray):
        raw = raw.flat[0]
    if isinstance(raw, bytes):
        raw = raw.decode()
    return [int(x) for x in raw.strip(",").split(",") if x.strip()]


def get_gene_record(f: h5py.File, idx: int) -> dict:
    """Read one gene's record from datafile_test_*.h5."""
    seq_full = f["SEQ"][idx]
    if isinstance(seq_full, bytes):
        seq_full = seq_full.rstrip(b"\x00").decode().upper()
    else:
        seq_full = seq_full.rstrip("\x00").upper()
    tx_start = int(f["TX_START"][idx])
    tx_end = int(f["TX_END"][idx])
    strand = f["STRAND"][idx].decode() if isinstance(f["STRAND"][idx], bytes) else f["STRAND"][idx]
    name = f["NAME"][idx].decode() if isinstance(f["NAME"][idx], bytes) else f["NAME"][idx]
    chrom = f["CHROM"][idx].decode() if isinstance(f["CHROM"][idx], bytes) else f["CHROM"][idx]
    paralog = int(f["PARALOG"][idx])
    jn_start = parse_int_csv(f["JN_START"][idx, 0])
    jn_end = parse_int_csv(f["JN_END"][idx, 0])
    gene_len = tx_end - tx_start
    # SpliceAI's original preprocessing layout (verified by decoding the X
    # tensors in dataset_test_0.h5):
    #   + strand: 5000 left + gene(gene_len) + 5001 right
    #   - strand: 5000 left + extended_fwd(gene_len + 1) + 5000 right
    # For - strand, the "extended" forward region includes ONE extra byte from
    # the right flank, which after reverse-complementing ends up at sense[0].
    # All sense junction positions then satisfy sense_pos = tx_end - g (no -1
    # offset), and sense_pos 0 is the extra byte (not part of the gene proper).
    # SpliceAI's original preprocessing uses gene_len + 1 bytes from the
    # forward strand: 5000 left + (gene_len+1) + 5000 right (total gene_len + 10001).
    # For + strand the extra byte sits AFTER the gene (sense[-1] is a flank byte).
    # For - strand the extra byte is the LAST forward byte, which after rev-comp
    # becomes sense[0] (a flank byte). The actual gene proper occupies:
    #   + strand: sense[0 : gene_len)
    #   - strand: sense[1 : 1 + gene_len)
    left_flank = 5000
    sense_len = gene_len + 1
    if strand == "+":
        gene_offset_in_sense = 0
    else:
        gene_offset_in_sense = 1
    right_flank = len(seq_full) - left_flank - sense_len
    if right_flank < 0 or len(seq_full) < sense_len:
        raise ValueError(f"Gene {idx} ({name}): seq length {len(seq_full)} < sense_len {sense_len}")
    return dict(
        name=name, chrom=chrom, strand=strand, tx_start=tx_start, tx_end=tx_end,
        paralog=paralog, gene_len=gene_len, flank=left_flank, right_flank=right_flank,
        sense_len=sense_len, gene_offset=gene_offset_in_sense,
        seq_full=seq_full, jn_start=jn_start, jn_end=jn_end,
    )


def gene_to_sense(rec: dict) -> tuple[str, list[int], list[int]]:
    """Return (sense_seq, donors_sense_sorted, acceptors_sense_sorted).

    sense_seq has length sense_len (= gene_len for +, gene_len+1 for -).
    For -, sense_seq[0] is one byte borrowed from the right flank in forward.

    Junction roles for - strand are swapped: JN_START = acceptor, JN_END = donor.
    """
    seq_full = rec["seq_full"]
    flank = rec["flank"]
    sense_len = rec["sense_len"]
    fwd = seq_full[flank: flank + sense_len]
    if rec["strand"] == "+":
        sense = fwd
        donors = sorted(g - rec["tx_start"] for g in rec["jn_start"])
        acceptors = sorted(g - rec["tx_start"] for g in rec["jn_end"])
    elif rec["strand"] == "-":
        sense = reverse_complement(fwd)
        # Empirically: sense_pos = tx_end - g (gives correct positions in
        # the gene_len+1 sense_seq). JN_START -> acceptor, JN_END -> donor.
        acceptors = sorted(rec["tx_end"] - g for g in rec["jn_start"])
        donors = sorted(rec["tx_end"] - g for g in rec["jn_end"])
    else:
        raise ValueError(f"Unknown strand {rec['strand']!r}")
    return sense, donors, acceptors


def build_exon_spans(donors: list[int], acceptors: list[int],
                     gene_start: int, gene_end_excl: int) -> list[tuple[int, int]]:
    """Half-open [start, end) spans of exons in sense-strand coords.

    Donor d is the LAST exon nucleotide; acceptor a is the FIRST exon nucleotide
    after the intron. ``gene_start`` is the first sense-position belonging to the
    gene (0 for + strand, 1 for - strand because sense[0] is a flank byte).
    ``gene_end_excl`` is the exclusive upper bound (gene_len for + strand,
    gene_len + 1 for - strand).
    """
    if len(donors) != len(acceptors):
        raise ValueError(f"donors/acceptors length mismatch: {len(donors)} vs {len(acceptors)}")
    if len(donors) == 0:
        return [(gene_start, gene_end_excl)]
    spans = [(gene_start, donors[0] + 1)]
    for i in range(1, len(donors)):
        spans.append((acceptors[i - 1], donors[i] + 1))
    spans.append((acceptors[-1], gene_end_excl))
    return spans


def motif_check(rec: dict) -> tuple[int, int, int]:
    """Return (canonical_donors, canonical_acceptors, total_pairs) for sanity."""
    sense, donors, acceptors = gene_to_sense(rec)
    n_donors_ok = 0
    n_acceptors_ok = 0
    for d in donors:
        if 0 <= d + 1 and d + 3 <= len(sense):
            m = sense[d + 1: d + 3]
            if m in ("GT", "GC", "AT"):
                n_donors_ok += 1
    for a in acceptors:
        if 2 <= a <= len(sense):
            m = sense[a - 2: a]
            if m in ("AG", "AC"):
                n_acceptors_ok += 1
    return n_donors_ok, n_acceptors_ok, len(donors)


def apply_nt_shuffle(sense: str, exons: list[tuple[int, int]], rng: np.random.Generator) -> str:
    arr = list(sense)
    pool = []
    for s, e in exons:
        pool.extend(arr[s:e])
    perm = rng.permutation(len(pool))
    shuffled = [pool[i] for i in perm]
    cursor = 0
    for s, e in exons:
        for i in range(s, e):
            arr[i] = shuffled[cursor]
            cursor += 1
    return "".join(arr)


def apply_codon_shuffle(sense: str, exons: list[tuple[int, int]], rng: np.random.Generator) -> str:
    arr = list(sense)
    pool: list[str] = []
    for s, e in exons:
        pool.extend(arr[s:e])
    n = len(pool)
    n_codons = n // 3
    rem = n - n_codons * 3
    if n_codons > 0:
        codons = [pool[3 * i: 3 * (i + 1)] for i in range(n_codons)]
        perm = rng.permutation(n_codons)
        new_pool: list[str] = []
        for i in perm:
            new_pool.extend(codons[i])
        if rem > 0:
            new_pool.extend(pool[3 * n_codons:])
    else:
        new_pool = pool
    cursor = 0
    for s, e in exons:
        for i in range(s, e):
            arr[i] = new_pool[cursor]
            cursor += 1
    return "".join(arr)


def apply_remove_k(
    sense: str,
    exons: list[tuple[int, int]],
    donors: list[int],
    acceptors: list[int],
    k: int,
    rng: np.random.Generator,
    gene_offset: int = 0,
) -> tuple[str, list[int], list[int], list[dict]]:
    """Delete k positions per exon. Returns (new_sense, new_donors, new_acceptors, deletions_meta).

    Deletions are uniform within each exon (any position eligible). Junctions
    shift via: new_pos = old_pos - count(deletions <= old_pos), which correctly
    handles the case where the donor position itself is deleted (the "new
    donor" merges onto the prior position).
    """
    delete_set: set[int] = set()
    deletions_meta = []
    for ei, (s, e) in enumerate(exons):
        exon_len = e - s
        n_del = min(k, exon_len)
        if n_del <= 0:
            continue
        chosen = rng.choice(np.arange(s, e), size=n_del, replace=False)
        for c in chosen:
            c = int(c)
            delete_set.add(c)
            denom = max(exon_len - 1, 1)
            deletions_meta.append({
                "exon_idx": ei,
                "exon_start": s,
                "exon_end": e,
                "abs_pos": c,
                "rel_in_exon": c - s,
                "frac_in_exon": (c - s) / denom,
            })

    sorted_dels = sorted(delete_set)
    # Preserve the leading flank byte for - strand (gene_offset=1): only delete
    # within the gene portion. Iterating over the entire sense and skipping
    # deleted indices does the right thing because chosen positions are within
    # exons (which lie in [gene_offset, gene_end_excl)).
    new_chars = [c for i, c in enumerate(sense) if i not in delete_set]
    new_sense = "".join(new_chars)

    def shift(pos: int) -> int:
        # count of deletions with del_pos <= pos
        # bisect_right gives count of elements <= pos
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
    """Convert manipulated sense sequence + junctions back to forward-strand
    bytes and genomic coords for storage in the datafile schema.

    Keeps tx_start fixed; new_tx_end = tx_start + new_gene_len. For - strand,
    the sense sequence is one byte longer than the gene proper because of the
    leading borrowed flank byte at sense[0]; that byte is preserved across
    manipulations and ends up at the END of the forward extended-gene region.

    Returns (new_extended_gene_fwd, new_jn_start_genomic, new_jn_end_genomic,
    new_tx_end). new_extended_gene_fwd has length new_gene_len for + strand
    and new_gene_len + 1 for - strand.
    """
    # sense_len is gene_len + 1 for both strands. After manipulation,
    # new_sense_len == sense_len for shuffle (no length change) or
    # sense_len - n_deletions for remove. new_gene_len = new_sense_len - 1.
    new_gene_len = len(new_sense) - 1
    new_tx_end = rec["tx_start"] + new_gene_len
    if rec["strand"] == "+":
        new_gene_fwd = new_sense
        new_jn_start = sorted(rec["tx_start"] + d for d in new_donors)      # JN_START = donor
        new_jn_end = sorted(rec["tx_start"] + a for a in new_acceptors)     # JN_END = acceptor
    else:
        new_gene_fwd = reverse_complement(new_sense)
        new_jn_start = sorted(new_tx_end - a for a in new_acceptors)        # JN_START = acceptor
        new_jn_end = sorted(new_tx_end - d for d in new_donors)             # JN_END = donor
    return new_gene_fwd, new_jn_start, new_jn_end, new_tx_end


def manipulate_gene(rec: dict, mode: str, rng: np.random.Generator) -> dict:
    sense, donors, acceptors = gene_to_sense(rec)
    gene_end_excl = rec["gene_offset"] + rec["gene_len"]
    exons = build_exon_spans(donors, acceptors, rec["gene_offset"], gene_end_excl)
    deletions_meta = []
    if mode == "nt_shuffle":
        new_sense = apply_nt_shuffle(sense, exons, rng)
        new_donors, new_acceptors = donors, acceptors
    elif mode == "codon_shuffle":
        new_sense = apply_codon_shuffle(sense, exons, rng)
        new_donors, new_acceptors = donors, acceptors
    elif mode in ("remove1", "remove2"):
        k = 1 if mode == "remove1" else 2
        new_sense, new_donors, new_acceptors, deletions_meta = apply_remove_k(
            sense, exons, donors, acceptors, k, rng, gene_offset=rec["gene_offset"]
        )
    else:
        raise ValueError(f"unknown mode {mode!r}")

    new_gene_fwd, new_jn_start, new_jn_end, new_tx_end = sense_to_forward_and_junctions(
        rec, new_sense, new_donors, new_acceptors
    )
    # Reassemble the SEQ field with original flanks (untouched).
    seq_full = rec["seq_full"]
    left = seq_full[:rec["flank"]]
    right = seq_full[rec["flank"] + rec["sense_len"]:]
    new_seq_full = left + new_gene_fwd + right

    # Sanity assertions
    if mode in ("nt_shuffle", "codon_shuffle"):
        orig_exon = "".join(sense[s:e] for s, e in exons)
        new_exon = "".join(new_sense[s:e] for s, e in exons)
        assert sorted(orig_exon) == sorted(new_exon), "exon composition not preserved"
        # Non-exonic positions (introns + the leading flank byte for - strand)
        # must be byte-identical to the source.
        non_exonic_mask = [True] * len(sense)
        for s, e in exons:
            for i in range(s, e):
                non_exonic_mask[i] = False
        for i, keep in enumerate(non_exonic_mask):
            if keep:
                assert sense[i] == new_sense[i], f"non-exonic byte changed at {i}"
        assert len(new_sense) == len(sense), "length changed"
    elif mode in ("remove1", "remove2"):
        k = 1 if mode == "remove1" else 2
        expected_dels = sum(min(k, e - s) for s, e in exons)
        assert len(new_sense) == len(sense) - expected_dels, \
            f"unexpected length: {len(new_sense)} != {len(sense)} - {expected_dels}"

    return dict(
        name=rec["name"],
        chrom=rec["chrom"],
        strand=rec["strand"],
        paralog=rec["paralog"],
        tx_start=rec["tx_start"],
        new_tx_end=new_tx_end,
        new_seq_full=new_seq_full,
        new_jn_start=new_jn_start,
        new_jn_end=new_jn_end,
        new_gene_len=new_tx_end - rec["tx_start"],
        old_gene_len=rec["gene_len"],
        deletions=deletions_meta,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", required=True, choices=["nt_shuffle", "codon_shuffle", "remove1", "remove2"])
    p.add_argument("--in", dest="in_path", required=True, help="Path to source datafile_test_0.h5")
    p.add_argument("--out", required=True, help="Path to write manipulated datafile")
    p.add_argument("--seed", type=int, default=None, help="Override default per-mode seed")
    p.add_argument("--limit", type=int, default=None, help="Process only first N genes (debugging)")
    p.add_argument("--manifest", default=None, help="Sidecar JSON path (default: <out>.manifest.json)")
    args = p.parse_args()

    base_seed = args.seed if args.seed is not None else DEFAULT_SEEDS[args.mode]
    print(f"Mode: {args.mode}  base_seed: {base_seed}")
    print(f"Input: {args.in_path}")
    print(f"Output: {args.out}")

    src = h5py.File(args.in_path, "r")
    n_genes = src["NAME"].shape[0]
    if args.limit is not None:
        n_genes = min(n_genes, args.limit)

    # Strand sanity check on a sample of 10 random genes before doing real work.
    sanity_rng = np.random.default_rng(0)
    sample_idx = sanity_rng.choice(src["NAME"].shape[0], size=min(10, src["NAME"].shape[0]), replace=False)
    total_d_ok = 0
    total_a_ok = 0
    total_pairs = 0
    for i in sample_idx:
        rec = get_gene_record(src, int(i))
        d_ok, a_ok, n_pairs = motif_check(rec)
        total_d_ok += d_ok
        total_a_ok += a_ok
        total_pairs += n_pairs
    if total_pairs > 0:
        d_frac = total_d_ok / total_pairs
        a_frac = total_a_ok / total_pairs
        print(f"Sanity check: donor canonical={d_frac:.3f} ({total_d_ok}/{total_pairs}), "
              f"acceptor canonical={a_frac:.3f} ({total_a_ok}/{total_pairs})")
        if d_frac < 0.8 or a_frac < 0.8:
            print("ERROR: canonical motif fraction below 0.8 — strand convention is wrong.",
                  file=sys.stderr)
            sys.exit(1)
    else:
        print("Sanity check: no junction pairs in sample (all single-exon?)")

    results = []
    new_seq_max = 0
    for i in range(n_genes):
        rec = get_gene_record(src, i)
        rng = np.random.default_rng(base_seed + i)
        out = manipulate_gene(rec, args.mode, rng)
        results.append(out)
        if len(out["new_seq_full"]) > new_seq_max:
            new_seq_max = len(out["new_seq_full"])
        if (i + 1) % 200 == 0 or i + 1 == n_genes:
            print(f"  processed {i+1}/{n_genes}")

    src.close()

    # Write the new datafile
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    seq_dtype = f"S{new_seq_max}"
    with h5py.File(out_path, "w") as dst:
        n = len(results)
        dst.create_dataset("NAME", data=np.array([r["name"].encode() for r in results], dtype="S16"))
        dst.create_dataset("CHROM", data=np.array([r["chrom"].encode() for r in results], dtype="S4"))
        dst.create_dataset("STRAND", data=np.array([r["strand"].encode() for r in results], dtype="S1"))
        dst.create_dataset("TX_START", data=np.array([r["tx_start"] for r in results], dtype=np.int64))
        dst.create_dataset("TX_END", data=np.array([r["new_tx_end"] for r in results], dtype=np.int64))
        dst.create_dataset("PARALOG", data=np.array([r["paralog"] for r in results], dtype=np.int64))

        seq_arr = np.array([r["new_seq_full"].encode() for r in results], dtype=seq_dtype)
        dst.create_dataset("SEQ", data=seq_arr)

        # JN_START / JN_END as comma-separated bytes inside (n,1) object arrays
        dt = h5py.special_dtype(vlen=bytes)
        jn_start_arr = np.empty((n, 1), dtype=object)
        jn_end_arr = np.empty((n, 1), dtype=object)
        for i, r in enumerate(results):
            jn_start_arr[i, 0] = (",".join(str(x) for x in r["new_jn_start"])).encode()
            jn_end_arr[i, 0] = (",".join(str(x) for x in r["new_jn_end"])).encode()
        dst.create_dataset("JN_START", data=jn_start_arr, dtype=dt)
        dst.create_dataset("JN_END", data=jn_end_arr, dtype=dt)

    # Manifest sidecar (deletion locations etc.)
    manifest_path = Path(args.manifest) if args.manifest else out_path.with_suffix(out_path.suffix + ".manifest.json")
    manifest = {
        "mode": args.mode,
        "seed": base_seed,
        "source": str(args.in_path),
        "n_genes": len(results),
        "genes": [
            {
                "name": r["name"],
                "strand": r["strand"],
                "old_gene_len": r["old_gene_len"],
                "new_gene_len": r["new_gene_len"],
                "deletions": r["deletions"],
            }
            for r in results
        ],
    }
    with open(manifest_path, "w") as fp:
        json.dump(manifest, fp, indent=2)
    print(f"Wrote datafile: {out_path}")
    print(f"Wrote manifest: {manifest_path}")


if __name__ == "__main__":
    main()
