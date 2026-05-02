"""
Build a per-gene CDS mask for the SpliceAI test set, using GENCODE V46-lift37.

For every gene in datafile_test_0.h5:
1. Find the GENCODE protein-coding transcript whose splice junctions match the
   datafile's junctions (with ±1 bp tolerance, then per-gene offset detection).
2. Convert that transcript's CDS intervals to sense-strand coordinates aligned
   with our datafile.
3. Validate the CDS:
     - the first three sense bytes of the CDS must be 'ATG'
     - total CDS length must be a multiple of 3
4. Write the per-gene CDS mask + validation outcome to JSON; log the excluded
   genes (with reason) to a separate JSON.

Usage:
    python experiments/build_cds_mask.py \\
        --datafile datafile_test_0.h5 \\
        --gff data/gencode/chr1.gff3 \\
        --out data/cds_masks.json \\
        --invalid-log data/cds_invalid_log.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "experiments"))
from manipulate_exons import (  # noqa: E402
    build_exon_spans,
    gene_to_sense,
    get_gene_record,
)


JUNCTION_TOLERANCE = 1
MIN_JUNCTION_MATCH_FRAC = 0.95
MAX_OFFSET = 2


def parse_attrs(s: str) -> dict:
    out = {}
    for kv in s.strip().split(";"):
        if "=" in kv:
            k, v = kv.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def parse_gff(gff_path: Path) -> tuple[dict, dict]:
    transcripts: dict = {}
    by_gene: dict = defaultdict(list)
    with open(gff_path) as fh:
        for line in fh:
            if line.startswith("#") or not line.strip():
                continue
            cols = line.rstrip("\n").split("\t")
            if len(cols) < 9:
                continue
            _, _, feat, start, end, _, strand, _, attrs = cols
            if feat == "transcript":
                a = parse_attrs(attrs)
                tid = a.get("ID") or a["transcript_id"]
                rec = dict(
                    tx_id=tid,
                    gene_name=a.get("gene_name", ""),
                    tx_type=a.get("transcript_type", ""),
                    strand=strand,
                    tx_start=int(start),
                    tx_end=int(end),
                    tags=set(t.strip() for t in a.get("tag", "").split(",") if t.strip()),
                    exons=[],
                    cds=[],
                )
                transcripts[tid] = rec
                by_gene[rec["gene_name"]].append(rec)
            elif feat == "exon":
                a = parse_attrs(attrs)
                pid = a.get("Parent", "")
                if pid in transcripts:
                    transcripts[pid]["exons"].append((int(start), int(end)))
            elif feat == "CDS":
                a = parse_attrs(attrs)
                pid = a.get("Parent", "")
                if pid in transcripts:
                    transcripts[pid]["cds"].append((int(start), int(end)))
    for r in transcripts.values():
        r["exons"].sort()
        r["cds"].sort()
    return transcripts, by_gene


def transcript_junctions(tx: dict) -> tuple[set[int], set[int]]:
    exons = sorted(tx["exons"])
    donors = {e - 1 for _, e in exons[:-1]}
    acceptors = {s - 1 for s, _ in exons[1:]}
    return donors, acceptors


def fuzzy_match(a: set[int], b: set[int], tol: int) -> int:
    matches = 0
    for x in a:
        for d in range(-tol, tol + 1):
            if (x + d) in b:
                matches += 1
                break
    return matches


def detect_offset(tx: dict, df_donors: set[int], df_acceptors: set[int]) -> int:
    g_d, g_a = transcript_junctions(tx)
    best_off, best_score = 0, -1
    for off in range(-MAX_OFFSET, MAX_OFFSET + 1):
        m = (len({x + off for x in g_d} & df_donors) +
             len({x + off for x in g_a} & df_acceptors))
        if m > best_score:
            best_off, best_score = off, m
    return best_off


def pick_matching_transcript(cands: list, df_donors: set[int], df_acceptors: set[int]):
    if not cands:
        return None, (0, 0), 0
    pc = [r for r in cands if r["tx_type"] == "protein_coding"]
    if pc:
        cands = pc
    best = None
    for tx in cands:
        if not tx["cds"]:
            continue
        g_d, g_a = transcript_junctions(tx)
        m = fuzzy_match(df_donors, g_d, JUNCTION_TOLERANCE) + fuzzy_match(df_acceptors, g_a, JUNCTION_TOLERANCE)
        t = len(df_donors) + len(df_acceptors)
        if best is None or (m, t) > (best[1][0], best[1][1]):
            best = (tx, (m, t))
    if best is None:
        return None, (0, 0), 0
    tx, (m, t) = best
    off = detect_offset(tx, df_donors, df_acceptors)
    return tx, (m, t), off


def gff_intervals_to_sense(rec: dict, gff_intervals: list[tuple[int, int]],
                           offset: int) -> list[tuple[int, int]]:
    out = []
    for s_1, e_1 in gff_intervals:
        g_lo, g_hi = s_1 - 1 + offset, e_1 + offset
        if rec["strand"] == "+":
            s_sense = g_lo - rec["tx_start"]
            e_sense = g_hi - rec["tx_start"]
        else:
            s_sense = rec["tx_end"] - g_hi + 1
            e_sense = rec["tx_end"] - g_lo + 1
        out.append((max(s_sense, 0), min(e_sense, rec["sense_len"])))
    return [iv for iv in sorted(out) if iv[1] > iv[0]]


def cds_within_exons(cds_intervals: list[tuple[int, int]],
                     exon_spans: list[tuple[int, int]]) -> bool:
    """All CDS positions must lie inside an exon span."""
    for s, e in cds_intervals:
        covered = False
        for es, ee in exon_spans:
            if s >= es and e <= ee:
                covered = True
                break
        if not covered:
            return False
    return True


def cds_sequence(sense: str, cds_intervals: list[tuple[int, int]]) -> str:
    return "".join(sense[s:e] for s, e in cds_intervals)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datafile", default="datafile_test_0.h5")
    ap.add_argument("--gff", default="data/gencode/chr1.gff3")
    ap.add_argument("--out", default="data/cds_masks.json")
    ap.add_argument("--invalid-log", default="data/cds_invalid_log.json")
    args = ap.parse_args()

    print("Parsing GFF…")
    transcripts, by_gene = parse_gff(Path(args.gff))
    print(f"  {len(transcripts):,} transcripts across {len(by_gene):,} genes")

    masks: dict[str, dict] = {}
    invalid: list[dict] = []

    with h5py.File(args.datafile, "r") as fh:
        n = fh["NAME"].shape[0]
        for i in range(n):
            rec = get_gene_record(fh, i)
            name = rec["name"]
            cands = by_gene.get(name, [])
            if not cands:
                invalid.append({"gene_idx": i, "name": name, "reason": "no_transcript_in_gencode"})
                continue
            df_donors = set(rec["jn_start"])
            df_acceptors = set(rec["jn_end"])
            tx, (m, t), off = pick_matching_transcript(cands, df_donors, df_acceptors)
            if tx is None:
                invalid.append({"gene_idx": i, "name": name, "reason": "no_protein_coding_match"})
                continue
            if t == 0 or m / t < MIN_JUNCTION_MATCH_FRAC:
                invalid.append({"gene_idx": i, "name": name,
                                "reason": f"low_junction_match: {m}/{t}",
                                "tx_id": tx["tx_id"]})
                continue
            cds_sense = gff_intervals_to_sense(rec, tx["cds"], off)
            sense, donors, acceptors = gene_to_sense(rec)
            exon_spans = build_exon_spans(donors, acceptors,
                                          rec["gene_offset"],
                                          rec["gene_offset"] + rec["gene_len"])
            if not cds_within_exons(cds_sense, exon_spans):
                invalid.append({"gene_idx": i, "name": name,
                                "reason": "cds_outside_exons",
                                "tx_id": tx["tx_id"]})
                continue
            cds_seq = cds_sequence(sense, cds_sense)
            cds_len = len(cds_seq)
            if cds_len < 3:
                invalid.append({"gene_idx": i, "name": name,
                                "reason": f"cds_too_short: {cds_len}",
                                "tx_id": tx["tx_id"]})
                continue
            starts_atg = cds_seq[:3].upper() == "ATG"
            mod3 = (cds_len % 3 == 0)
            if not (starts_atg and mod3):
                reason = []
                if not starts_atg:
                    reason.append(f"no_atg_start: cds[:3]={cds_seq[:3]}")
                if not mod3:
                    reason.append(f"not_mod3: cds_len={cds_len}")
                invalid.append({"gene_idx": i, "name": name,
                                "reason": "; ".join(reason),
                                "tx_id": tx["tx_id"],
                                "cds_len": cds_len})
                continue

            masks[str(i)] = {
                "name": name,
                "strand": rec["strand"],
                "tx_id": tx["tx_id"],
                "gencode_offset": off,
                "junction_match": [m, t],
                "cds_sense_intervals": cds_sense,
                "cds_total_length": cds_len,
            }

            if (i + 1) % 200 == 0 or i + 1 == n:
                print(f"  processed {i+1}/{n}  valid={len(masks)}  invalid={len(invalid)}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(masks, indent=2))
    Path(args.invalid_log).parent.mkdir(parents=True, exist_ok=True)
    Path(args.invalid_log).write_text(json.dumps(invalid, indent=2))

    print(f"\nValid:   {len(masks):,} genes  → {out_path}")
    print(f"Invalid: {len(invalid):,} genes  → {args.invalid_log}")
    reason_counts: dict[str, int] = {}
    for entry in invalid:
        key = entry["reason"].split(":")[0]
        reason_counts[key] = reason_counts.get(key, 0) + 1
    print("Reasons:")
    for k, v in sorted(reason_counts.items(), key=lambda kv: -kv[1]):
        print(f"  {k:35s} {v:>5,}")


if __name__ == "__main__":
    main()
