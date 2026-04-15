"""
Prepare tissue-specific splice site labels from GTEx v8 junction counts.

Uses the publicly available GTEx junction GCT file:
  GTEx_Analysis_2017-06-05_v8_STARv2.5.3a_junctions.gct.gz

Combined with GTEx sample metadata to map samples to tissues, this creates
binary splice site labels per tissue for the chr1 test genes.

Usage:
    python prepare_gtex_labels.py
    python prepare_gtex_labels.py --min-reads 5 --min-samples 2
"""

from __future__ import annotations

import argparse
import gzip
import sys
import urllib.request
from collections import defaultdict
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parent.parent)

import h5py
import numpy as np
from pyliftover import LiftOver

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

JUNCTION_GCT = str(Path(_REPO_ROOT) / "GTEx_Analysis_2017-06-05_v8_STARv2.5.3a_junctions.gct.gz")

GTEX_SAMPLE_ATTRS_URL = (
    "https://storage.googleapis.com/adult-gtex/annotations/v8/"
    "metadata-files/GTEx_Analysis_v8_Annotations_SampleAttributesDS.txt"
)

# Tissues we care about — map to GTEx SMTSD (tissue detail) names
TISSUES = {
    "heart": [
        "Heart - Left Ventricle",
        "Heart - Atrial Appendage",
    ],
    "liver": ["Liver"],
    "brain": [
        "Brain - Cerebellum",
        "Brain - Cerebellar Hemisphere",
    ],
    "testis": ["Testis"],
}


# ---------------------------------------------------------------------------
# Step 1: Map GTEx sample IDs to tissues
# ---------------------------------------------------------------------------

def load_sample_tissue_map(cache_dir: Path) -> dict[str, str]:
    """Download GTEx sample annotations, return sample_id -> tissue_key.

    Only returns samples belonging to our 4 target tissues.
    """
    cache_path = cache_dir / "GTEx_SampleAttributes.txt"

    if not cache_path.exists():
        print("  Downloading GTEx sample metadata...")
        urllib.request.urlretrieve(GTEX_SAMPLE_ATTRS_URL, cache_path)
        print(f"  Saved to {cache_path}")
    else:
        print(f"  Using cached metadata: {cache_path}")

    # Build reverse map: SMTSD detail name -> our tissue key
    detail_to_key = {}
    for tissue_key, detail_names in TISSUES.items():
        for d in detail_names:
            detail_to_key[d] = tissue_key

    sample_to_tissue = {}
    tissue_counts = defaultdict(int)

    with open(cache_path) as f:
        header = f.readline().strip().split("\t")
        sampid_idx = header.index("SAMPID")
        smtsd_idx = header.index("SMTSD")

        for line in f:
            fields = line.strip().split("\t")
            if len(fields) <= max(sampid_idx, smtsd_idx):
                continue
            sampid = fields[sampid_idx]
            tissue_detail = fields[smtsd_idx]

            tissue_key = detail_to_key.get(tissue_detail)
            if tissue_key:
                sample_to_tissue[sampid] = tissue_key
                tissue_counts[tissue_key] += 1

    for t in sorted(tissue_counts):
        print(f"    {t}: {tissue_counts[t]} samples")

    return sample_to_tissue


# ---------------------------------------------------------------------------
# Step 2: Parse junction GCT for chr1, aggregate per tissue
# ---------------------------------------------------------------------------

def parse_chr1_junctions(
    gct_path: str,
    sample_to_tissue: dict[str, str],
    min_reads: int = 5,
    min_samples: int = 2,
) -> dict[str, set[int]]:
    """Parse the GTEx junction GCT, extract chr1 junctions per tissue.

    For each tissue, a splice site (donor or acceptor position) is "active"
    if the junction has >= min_reads in >= min_samples from that tissue.

    GTEx v8 uses hg38 coordinates, but SpliceAI's datafiles use hg19.
    We liftOver each junction coordinate from hg38 -> hg19.

    Returns: tissue -> set of active splice site positions on chr1 (hg19).
    """
    print(f"\n  Parsing {gct_path} (chr1 junctions only)...")
    print("  Loading hg38 -> hg19 liftOver chain...")
    lo = LiftOver('hg38', 'hg19')

    def to_hg19(pos: int) -> int | None:
        """Convert a single hg38 chr1 position to hg19. Returns None if unmapped.

        GCT junction coordinates are 1-based (STAR output).
        pyliftover expects 0-based input, so subtract 1 before converting.
        The datafile coordinates (hg19) are also 0-based.
        """
        result = lo.convert_coordinate('chr1', pos - 1)  # 1-based -> 0-based
        if result and len(result) == 1 and result[0][0] == 'chr1':
            return result[0][1]
        return None

    with gzip.open(gct_path, "rt") as f:
        # Skip GCT header lines
        f.readline()  # #1.2
        dims = f.readline().strip().split("\t")  # n_junctions  n_samples
        n_junctions, n_samples = int(dims[0]), int(dims[1])
        print(f"    {n_junctions} junctions × {n_samples} samples")

        # Parse column header to map column indices to tissues
        header = f.readline().strip().split("\t")
        # Columns: Name, Description, sample1, sample2, ...
        sample_ids = header[2:]

        # Build tissue -> list of column indices (0-based within count columns)
        tissue_col_indices = defaultdict(list)
        for col_idx, sid in enumerate(sample_ids):
            tissue = sample_to_tissue.get(sid)
            if tissue:
                tissue_col_indices[tissue].append(col_idx)

        for t in sorted(tissue_col_indices):
            print(f"    {t}: {len(tissue_col_indices[t])} sample columns matched")

        # Stream data rows, only keep chr1
        tissue_sites = {t: set() for t in TISSUES}
        n_chr1 = 0
        n_processed = 0

        for line in f:
            n_processed += 1
            if n_processed % 50000 == 0:
                print(f"    Processed {n_processed}/{n_junctions} junctions "
                      f"({n_chr1} chr1)...")

            # Quick check: only parse chr1 lines
            if not line.startswith("chr1_"):
                continue

            fields = line.strip().split("\t")
            junction_name = fields[0]  # e.g. "chr1_12058_12178"

            parts = junction_name.split("_")
            donor_hg38 = int(parts[1])
            acceptor_hg38 = int(parts[2])

            counts = fields[2:]  # read counts per sample
            n_chr1 += 1

            for tissue, col_indices in tissue_col_indices.items():
                # Count samples with >= min_reads for this junction
                n_above = 0
                for ci in col_indices:
                    if ci < len(counts):
                        try:
                            if int(counts[ci]) >= min_reads:
                                n_above += 1
                                if n_above >= min_samples:
                                    break
                        except ValueError:
                            continue

                if n_above >= min_samples:
                    # LiftOver hg38 -> hg19
                    donor_hg19 = to_hg19(donor_hg38)
                    acceptor_hg19 = to_hg19(acceptor_hg38)
                    if donor_hg19 is not None:
                        tissue_sites[tissue].add(donor_hg19)
                    if acceptor_hg19 is not None:
                        # STAR reports last intron base; annotation marks
                        # first exon base (intron_end + 2, past the AG).
                        tissue_sites[tissue].add(acceptor_hg19 + 2)

    print(f"\n    Total chr1 junctions: {n_chr1}")
    for t in sorted(tissue_sites):
        print(f"    {t}: {len(tissue_sites[t])} active splice sites")

    return tissue_sites


# ---------------------------------------------------------------------------
# Step 3: Map splice sites to test gene windows
# ---------------------------------------------------------------------------

def create_window_labels(
    tissue_sites: dict[str, set[int]],
    datafile_path: str,
) -> dict[str, np.ndarray]:
    """Map tissue-specific splice sites to test gene window coordinates.

    Returns: tissue -> (total_windows, 5000) binary int8 labels.
    """
    with h5py.File(datafile_path, "r") as f:
        tx_starts = f["TX_START"][:]
        tx_ends = f["TX_END"][:]
        n_genes = len(tx_starts)

    windows_per_gene = np.ceil((tx_ends - tx_starts) / 5000).astype(np.int64)
    total_windows = int(windows_per_gene.sum())

    print(f"\n  Mapping to {total_windows} windows across {n_genes} genes...")

    tissue_labels = {}
    for tissue, sites in tissue_sites.items():
        labels = np.zeros((total_windows, 5000), dtype=np.int8)
        win_offset = 0
        n_mapped = 0

        for g_idx in range(n_genes):
            n_win = int(windows_per_gene[g_idx])
            tx_start = int(tx_starts[g_idx])

            for w in range(n_win):
                win_genomic_start = tx_start + w * 5000
                win_genomic_end = win_genomic_start + 5000
                for site in sites:
                    if win_genomic_start <= site < win_genomic_end:
                        pos_in_window = site - win_genomic_start
                        labels[win_offset + w, pos_in_window] = 1
                        n_mapped += 1

            win_offset += n_win

        tissue_labels[tissue] = labels
        print(f"    {tissue}: {n_mapped} positions mapped to windows")

    return tissue_labels


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def prepare_gtex_labels(
    gct_path: str = JUNCTION_GCT,
    datafile_path: str = str(Path(_REPO_ROOT) / "datafile_test_0.h5"),
    output_path: str = "gtex_tissue_labels_chr1.h5",
    cache_dir: str = "data_cache",
    min_reads: int = 5,
    min_samples: int = 2,
):
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("PREPARING TISSUE-SPECIFIC SPLICE SITE LABELS")
    print("=" * 60)

    # Step 1: Sample -> tissue mapping
    print("\nStep 1: Loading sample-tissue mapping...")
    sample_to_tissue = load_sample_tissue_map(cache)

    # Step 2: Parse chr1 junctions from GCT
    print("\nStep 2: Parsing GTEx junction counts...")
    tissue_sites = parse_chr1_junctions(
        gct_path, sample_to_tissue,
        min_reads=min_reads, min_samples=min_samples,
    )

    # Step 3: Map to window coordinates
    print("\nStep 3: Creating window-level labels...")
    tissue_labels = create_window_labels(tissue_sites, datafile_path)

    # Save
    out = Path(output_path)
    print(f"\nSaving to {out}...")
    with h5py.File(out, "w") as f:
        for tissue, labels in tissue_labels.items():
            f.create_dataset(
                f"{tissue}_labels", data=labels,
                compression="gzip", compression_opts=4,
            )
            n_splice = int(labels.sum())
            n_total = labels.size
            print(f"  {tissue}: {n_splice} splice positions / "
                  f"{n_total} total ({n_splice/n_total:.6%})")

    print(f"\nDone! Saved to {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare tissue-specific splice labels from GTEx junctions"
    )
    parser.add_argument(
        "--gct", type=str, default=JUNCTION_GCT,
        help=f"GTEx junction GCT file (default: {JUNCTION_GCT})",
    )
    parser.add_argument(
        "--datafile", type=str,
        default=str(Path(_REPO_ROOT) / "datafile_test_0.h5"),
        help="Test gene metadata HDF5",
    )
    parser.add_argument(
        "--output", type=str,
        default=str(Path(_REPO_ROOT) / "gtex_tissue_labels_chr1.h5"),
        help="Output HDF5 path",
    )
    parser.add_argument(
        "--cache-dir", type=str, default="data_cache",
        help="Cache directory for metadata downloads",
    )
    parser.add_argument(
        "--min-reads", type=int, default=5,
        help="Min reads per sample to count a junction (default: 5)",
    )
    parser.add_argument(
        "--min-samples", type=int, default=2,
        help="Min samples with reads to call a junction active (default: 2)",
    )
    args = parser.parse_args()

    prepare_gtex_labels(
        gct_path=args.gct,
        datafile_path=args.datafile,
        output_path=args.output,
        cache_dir=args.cache_dir,
        min_reads=args.min_reads,
        min_samples=args.min_samples,
    )
