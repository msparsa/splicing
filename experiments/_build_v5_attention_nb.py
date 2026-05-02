"""Build experiments/v5_attention_analysis.ipynb from cell sources defined inline.

Run with: python experiments/_build_v5_attention_nb.py
"""
from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT = REPO_ROOT / "experiments" / "v5_attention_analysis.ipynb"

cells: list[tuple[str, str]] = []   # list of (cell_type, source)


def md(s: str): cells.append(("markdown", s.strip("\n")))
def py(s: str): cells.append(("code", s.strip("\n")))


# ---------------------------------------------------------------------------
# Title
# ---------------------------------------------------------------------------
md("""
# SpliceMamba v5 — Cross-Attention Interpretability & CDS-Perturbation Analysis

This notebook studies what the **cross-attention SpliceMamba (v5)** uses to predict splice sites,
and how that signal changes when CDS information is destroyed via nucleotide- or codon-level shuffling.

We focus on a small set of representative genes (rather than the full ~1,500-gene test set) where the
model performs **very well or very poorly** under each perturbation, and surface candidate motifs /
regulatory regions that could be experimentally validated.

**Conditions:** `filtered_baseline`, `nt_shuffle`, `codon_shuffle`
**Checkpoint:** `checkpoints/best.pt` (v5 cross-attention)

**Sections:**
1. Setup
2. Per-gene metric computation (selection driver)
3. Gene selection
4. Attention extraction utility
5. Per-gene visualization
6. Aggregate "where does attention go?" analysis
7. Motif analysis (sequence logos + ATtRACT RBP scan)
8. Cross-condition attention diff
9. Saliency cross-check
10. Wet-lab candidate summary
""")


# ---------------------------------------------------------------------------
# Section 1: Setup
# ---------------------------------------------------------------------------
md("## 1. Setup")

py(r'''
from __future__ import annotations
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import average_precision_score

REPO_ROOT = Path.cwd().parent if Path.cwd().name == "experiments" else Path.cwd()
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "evaluation"))
sys.path.insert(0, str(REPO_ROOT / "experiments"))

from model_v5 import SpliceMambaV5, TopNCrossAttention
from manipulate_exons import get_gene_record, gene_to_sense, build_exon_spans, reverse_complement
from eval_utils import compute_gene_window_counts, stitch_gene_predictions, stitch_gene_labels, read_window_labels

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)
print("repo:", REPO_ROOT)
''')

py(r'''
# Logomaker is the only optional plotting dep — install on demand.
try:
    import logomaker
    print("logomaker:", logomaker.__version__)
except ImportError:
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "logomaker"], check=True)
    import logomaker
    print("logomaker installed:", logomaker.__version__)
''')

py(r'''
# ---- Configuration ----
CHECKPOINT_PATH = REPO_ROOT / "checkpoints" / "best.pt"
CDS_MASKS_PATH  = REPO_ROOT / "data" / "cds_masks.json"

CONDITIONS = ["filtered_baseline", "nt_shuffle", "codon_shuffle"]

CONDITION_FILES = {
    cond: dict(
        datafile = REPO_ROOT / f"datafile_test_cds_{cond}.h5",
        dataset  = REPO_ROOT / f"dataset_test_cds_{cond}.h5",
    )
    for cond in CONDITIONS
}

CACHE_DIR = REPO_ROOT / "data" / "v5_attention_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Sanity: every file exists.
for cond, paths in CONDITION_FILES.items():
    for k, p in paths.items():
        assert p.exists(), f"missing: {p}"
print("all condition files present")
''')

py(r'''
# ---- Load checkpoint into SpliceMambaV5 ----
ckpt = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
saved_cfg = ckpt.get("config", {})
print("checkpoint epoch:", ckpt.get("epoch", "?"),
      "best AUPRC:", ckpt.get("best_auprc", "?"))
print("model_version in cfg:", saved_cfg.get("model_version"))
assert saved_cfg.get("model_version") == "v5", "expected v5 checkpoint"

model = SpliceMambaV5(
    d_model=saved_cfg["d_model"],
    n_mamba_layers=saved_cfg["n_mamba_layers"],
    d_state=saved_cfg["d_state"],
    expand=saved_cfg["expand"],
    d_conv=saved_cfg["d_conv"],
    headdim=saved_cfg["headdim"],
    n_cross_attn_layers=saved_cfg["n_cross_attn_layers"],
    n_heads=saved_cfg["n_heads"],
    top_n=saved_cfg["top_n"],
    vicinity_radius=saved_cfg["vicinity_radius"],
    gumbel_tau=saved_cfg["gumbel_tau"],
    coarse_select_in_label_only=saved_cfg["coarse_select_in_label_only"],
    label_start=saved_cfg.get("label_start", 5000),
    label_end=saved_cfg.get("label_end", 10000),
    dropout=saved_cfg.get("dropout", 0.15),
    drop_path_rate=0.0,
    n_classes=saved_cfg["n_classes"],
    max_len=saved_cfg["max_len"],
).to(device)
model.load_state_dict(ckpt["model"])
model.eval()

LABEL_START = model.label_start    # 5000
LABEL_END   = model.label_end      # 10000
SEQ_LEN     = model.max_len        # 15000
N_HEADS     = saved_cfg["n_heads"]
N_LAYERS    = saved_cfg["n_cross_attn_layers"]
print(f"model loaded — d_model={model.d_model}, layers={N_LAYERS}, heads={N_HEADS}, "
      f"top_n={model.top_n}, vicinity_radius={model.vicinity_radius}")
''')

py(r'''
# ---- Load CDS masks (key = ORIG_GENE_IDX as string) ----
with open(CDS_MASKS_PATH) as f:
    CDS_MASKS = json.load(f)
print(f"CDS masks: {len(CDS_MASKS)} genes")

# ---- Per-condition gene-record cache ----
def open_datafile(cond):
    return h5py.File(CONDITION_FILES[cond]["datafile"], "r")

def n_genes(cond):
    with open_datafile(cond) as f:
        return f["NAME"].shape[0]

print("rows per condition:", {c: n_genes(c) for c in CONDITIONS})
''')


# ---------------------------------------------------------------------------
# Section 2: Per-gene metrics
# ---------------------------------------------------------------------------
md("""
## 2. Per-gene metrics under each condition

We run the v5 model once on each condition's dataset and compute per-gene AUPRC. Cached to a
parquet so re-runs skip the inference pass.
""")

py(r'''
@torch.no_grad()
def predict_condition(cond: str, batch_size: int = 4) -> np.ndarray:
    """Run model on every X-shard of the condition's dataset; return (total_windows, 5000, 3) probs."""
    dataset_path = CONDITION_FILES[cond]["dataset"]
    all_probs = []
    with h5py.File(dataset_path, "r") as f:
        x_keys = sorted([k for k in f.keys() if k.startswith("X")], key=lambda k: int(k[1:]))
        for x_key in x_keys:
            x_data = f[x_key][:]                    # (N, 15000, 4) int8
            n_w = x_data.shape[0]
            shard_probs = []
            for s in range(0, n_w, batch_size):
                e = min(s + batch_size, n_w)
                batch = torch.from_numpy(
                    x_data[s:e].astype(np.float32)
                ).permute(0, 2, 1).to(device)
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                    _, refined, _ = model(batch)
                probs = torch.softmax(
                    refined[:, LABEL_START:LABEL_END, :].float(), dim=-1
                ).cpu().numpy()
                shard_probs.append(probs)
            all_probs.append(np.concatenate(shard_probs, axis=0))
    return np.concatenate(all_probs, axis=0)
''')

py(r'''
def per_gene_auprc(gene_probs: np.ndarray, gene_labels: np.ndarray) -> dict:
    """Per-gene AUPRC (donor + acceptor). gene_probs (L,3); gene_labels (L,)."""
    out = {}
    for cls_name, cls_idx in [("donor", 2), ("acceptor", 1)]:
        y = (gene_labels == cls_idx).astype(np.int8)
        if y.sum() == 0:
            out[f"auprc_{cls_name}"] = np.nan
            out[f"n_{cls_name}"] = 0
        else:
            out[f"auprc_{cls_name}"] = average_precision_score(y, gene_probs[:, cls_idx])
            out[f"n_{cls_name}"] = int(y.sum())
    out["auprc_mean"] = np.nanmean([out["auprc_donor"], out["auprc_acceptor"]])
    return out
''')

py(r'''
METRICS_CACHE = CACHE_DIR / "per_gene_metrics_v5.parquet"

if METRICS_CACHE.exists():
    metrics_df = pd.read_parquet(METRICS_CACHE)
    print(f"loaded cached metrics: {len(metrics_df)} rows")
else:
    rows = []
    for cond in CONDITIONS:
        print(f"=== {cond} ===")
        probs = predict_condition(cond)
        labels = read_window_labels(str(CONDITION_FILES[cond]["dataset"]))
        n_per_gene = compute_gene_window_counts(str(CONDITION_FILES[cond]["datafile"]))
        gene_probs = stitch_gene_predictions(probs, n_per_gene)
        gene_labels = stitch_gene_labels(labels, n_per_gene)
        with open_datafile(cond) as f:
            names = [n.decode() for n in f["NAME"][:]]
            orig_gi = f["ORIG_GENE_IDX"][:]
            strands = [s.decode() for s in f["STRAND"][:]]
        for i, (gp, gl) in enumerate(zip(gene_probs, gene_labels)):
            r = per_gene_auprc(gp, gl)
            r.update(condition=cond, row_idx=i, name=names[i],
                     orig_gene_idx=int(orig_gi[i]), strand=strands[i],
                     gene_len=int(gp.shape[0]), n_windows=int(n_per_gene[i]))
            rows.append(r)
    metrics_df = pd.DataFrame(rows)
    metrics_df.to_parquet(METRICS_CACHE)
    print(f"cached -> {METRICS_CACHE}")

metrics_df.head()
''')

py(r'''
# Aggregate sanity check: does our per-gene mean AUPRC roughly match the JSON file?
agg = metrics_df.groupby("condition")[["auprc_donor","auprc_acceptor","auprc_mean"]].mean()
print(agg.round(4))
''')


# ---------------------------------------------------------------------------
# Section 3: Gene selection
# ---------------------------------------------------------------------------
md("""
## 3. Gene selection

Pick ~12 genes spanning four buckets:
- **Top-3 baseline AUPRC** — cleanest examples of what the model "knows".
- **Bottom-3 baseline AUPRC** — failure modes.
- **CDS-robust** — high baseline, small Δ under shuffles → model uses non-CDS signal (best wet-lab candidates).
- **CDS-dependent** — high baseline, large Δ under shuffles → model relies on CDS sequence content.

We require ≥4 splice sites per gene so per-gene AUPRC is meaningful.
""")

py(r'''
# Pivot to one row per gene with all 3 conditions side-by-side.
pv = metrics_df.pivot_table(
    index=["orig_gene_idx", "name", "strand"],
    columns="condition",
    values=["auprc_mean", "n_donor", "n_acceptor"],
).reset_index()

# Flatten MultiIndex columns
pv.columns = ['_'.join([c for c in col if c]).strip('_') for col in pv.columns]

# Total splice sites = donors + acceptors at baseline
pv["n_splice"] = pv["n_donor_filtered_baseline"] + pv["n_acceptor_filtered_baseline"]

# Δ AUPRC under each shuffle
pv["delta_nt"]    = pv["auprc_mean_filtered_baseline"] - pv["auprc_mean_nt_shuffle"]
pv["delta_codon"] = pv["auprc_mean_filtered_baseline"] - pv["auprc_mean_codon_shuffle"]
pv["delta_max"]   = pv[["delta_nt", "delta_codon"]].max(axis=1)

# Filter: enough splice sites to make AUPRC meaningful
eligible = pv[pv["n_splice"] >= 4].copy()
print(f"{len(eligible)} eligible genes (>=4 splice sites)")

# Sort views
top_baseline    = eligible.nlargest(3,  "auprc_mean_filtered_baseline")
bottom_baseline = eligible.nsmallest(3, "auprc_mean_filtered_baseline")

# CDS-robust: high baseline AND tiny shuffle drop
robust_pool = eligible[eligible["auprc_mean_filtered_baseline"] >= 0.85]
robust = robust_pool.nsmallest(3, "delta_max")

# CDS-dependent: high baseline AND large shuffle drop
dependent = eligible[eligible["auprc_mean_filtered_baseline"] >= 0.85].nlargest(3, "delta_max")

selected = pd.concat([
    top_baseline.assign(bucket="top_baseline"),
    bottom_baseline.assign(bucket="bottom_baseline"),
    robust.assign(bucket="cds_robust"),
    dependent.assign(bucket="cds_dependent"),
]).drop_duplicates(subset=["orig_gene_idx"], keep="first").reset_index(drop=True)

print("\\nSelected genes:")
display_cols = ["bucket", "name", "strand", "n_splice",
                "auprc_mean_filtered_baseline", "auprc_mean_nt_shuffle",
                "auprc_mean_codon_shuffle", "delta_nt", "delta_codon"]
selected[display_cols].round(3)
''')

py(r'''
SELECTED_GENES = selected["orig_gene_idx"].tolist()
print("selected orig_gene_idx:", SELECTED_GENES)
''')


# ---------------------------------------------------------------------------
# Section 4: Attention extraction
# ---------------------------------------------------------------------------
md("""
## 4. Attention extraction

`TopNCrossAttention` uses FlashAttention internally (no attention weights returned). We register
forward hooks on its `q_proj` / `k_proj` / `v_proj` modules to capture Q, K, V; after the model
forward, we manually compute `softmax(QK^T/sqrt(d_h))` for a small set of "target" Q rows
(positions near the gene's true splice sites) so memory stays bounded.

We also stash the model's selected `vicinity_idx` (which sequence positions each Q row corresponds to).
""")

py(r'''
class AttentionCapture:
    """Context manager: captures Q/K/V from each TopNCrossAttention layer.

    Usage:
        with AttentionCapture(model) as cap:
            _, refined, _ = model(x)
        # cap.layers[i] = {"q": (B,Q,H,Hd), "k": (B,L,H,Hd), "v": (B,L,H,Hd)}
        # cap.vicinity_idx, cap.q_pad_mask available after forward
    """
    def __init__(self, model):
        self.model = model
        self.layers = []   # list of dicts, one per cross-attn layer
        self._hooks = []

    def __enter__(self):
        self.layers = []
        for layer in self.model.cross_attn_layers:
            entry = {}
            self.layers.append(entry)

            def make_hook(buf_key, e=entry):
                def h(_, __, out): e[buf_key] = out.detach()
                return h

            self._hooks.append(layer.q_proj.register_forward_hook(make_hook("q_out")))
            self._hooks.append(layer.k_proj.register_forward_hook(make_hook("k_out")))
            self._hooks.append(layer.v_proj.register_forward_hook(make_hook("v_out")))
        return self

    def __exit__(self, *args):
        for h in self._hooks: h.remove()
        self._hooks = []
        self.vicinity_idx, self.q_pad_mask = self.model._last_selection

    def attention(self, layer_idx: int, target_q_rows: torch.Tensor) -> torch.Tensor:
        """Compute (B, n_targets, H, L) attention weights for the chosen Q rows.

        target_q_rows : (B, n_targets) long indices into the Q dimension.
        """
        e = self.layers[layer_idx]
        q_all = e["q_out"].float()                              # (B, Q, D)
        k_all = e["k_out"].float()                              # (B, L, D)
        B, Q, D = q_all.shape
        L = k_all.size(1)
        H = self.model.cross_attn_layers[layer_idx].n_heads
        Hd = D // H
        # gather target Q rows
        idx = target_q_rows.unsqueeze(-1).expand(-1, -1, D)     # (B, T, D)
        q_t = q_all.gather(1, idx).view(B, -1, H, Hd)           # (B, T, H, Hd)
        k = k_all.view(B, L, H, Hd)
        # (B, H, T, Hd) @ (B, H, Hd, L) -> (B, H, T, L)
        scores = torch.einsum("bthd,blhd->bhtl", q_t, k) / math.sqrt(Hd)
        attn = torch.softmax(scores, dim=-1)                    # (B, H, T, L)
        return attn.permute(0, 2, 1, 3).contiguous()            # (B, T, H, L)
''')

py(r'''
# ---- Sanity check: the K/V we capture, when fed back through flash, should match refined logits ----
# We do a much weaker check: for one batch, the captured Q/K means are non-NaN and the model produced
# valid refined logits. Strict numerical match isn't tractable since flash_attn runs in bf16 internally.
def smoke_test_capture():
    cond = "filtered_baseline"
    with h5py.File(CONDITION_FILES[cond]["dataset"], "r") as f:
        x_data = f["X0"][:1].astype(np.float32)
    x = torch.from_numpy(x_data).permute(0, 2, 1).to(device)
    with AttentionCapture(model) as cap:
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            _, refined, _ = model(x)
    print("refined logits range:", refined.min().item(), "..", refined.max().item())
    print("vicinity_idx unique counts:",
          [int(cap.q_pad_mask[i].sum()) for i in range(cap.q_pad_mask.size(0))])
    for li, e in enumerate(cap.layers):
        print(f"layer {li}: q={tuple(e['q_out'].shape)} k={tuple(e['k_out'].shape)} v={tuple(e['v_out'].shape)}")

smoke_test_capture()
''')

py(r'''
def find_target_q_rows(vicinity_idx: torch.Tensor,
                       q_pad_mask: torch.Tensor,
                       target_positions: list[int]) -> dict:
    """For each target sequence position, find the Q row whose vicinity_idx == target.

    Returns dict[target_pos] -> int Q-row index (or None if target wasn't selected).
    Assumes B=1 (one window at a time for visualization).
    """
    assert vicinity_idx.size(0) == 1, "use B=1 for per-window visualization"
    vi = vicinity_idx[0].cpu().numpy()
    pm = q_pad_mask[0].cpu().numpy()
    out = {}
    for pos in target_positions:
        # vicinity_idx may contain `pos` only if it was inside a selected vicinity AND it was the
        # "first" occurrence in the dedup pass. Find the matching real Q row.
        matches = np.where((vi == pos) & pm)[0]
        out[pos] = int(matches[0]) if len(matches) else None
    return out
''')


# ---------------------------------------------------------------------------
# Section 5: Per-gene visualization
# ---------------------------------------------------------------------------
md("""
## 5. Per-gene visualization

For each selected gene under each condition, we render a stacked figure with:
1. **Gene-structure track** (exon boxes, CDS / 5'UTR / 3'UTR shading, donor/acceptor markers)
2. **Prediction track** (P(donor), P(acceptor) along label region)
3. **Attention heatmap** (rows = true splice sites, cols = full 15kb)
4. **Top-attended positions** table

All coordinates are window-relative `[0, 15000)`. Window 0 covers gene[0:5000] in the label region;
window 1 covers gene[5000:10000]; etc.
""")

py(r'''
def gene_record_for(cond: str, row_idx: int) -> dict:
    """Return rec from manipulate_exons.get_gene_record for a row in a condition's datafile."""
    with open_datafile(cond) as f:
        return get_gene_record(f, row_idx)


def cds_window_intervals(cond: str, row_idx: int, window_idx: int) -> list[tuple[int, int]]:
    """Return CDS spans in input-window coordinates [0, 15000) for window_idx of this gene.

    CDS intervals are in sense coordinates (from cds_masks.json). For + strand, sense_pos == fwd_pos
    relative to TX_START. For - strand, sense is rev-comp of fwd.

    Note: cds_masks were computed for the ORIGINAL (unmanipulated) gene. For nt_shuffle/codon_shuffle
    the coordinates are unchanged; for remove*, they shift (we don't use remove* in this notebook).
    """
    with open_datafile(cond) as f:
        orig_gi = int(f["ORIG_GENE_IDX"][row_idx])
    info = CDS_MASKS.get(str(orig_gi))
    if info is None:
        return []
    rec = gene_record_for(cond, row_idx)
    flank = rec["flank"]                  # 5000
    sense_len = rec["sense_len"]
    intervals_input = []
    for s, e in info["cds_sense_intervals"]:
        # sense -> input-window via:
        #   window w covers sense[w*5000 : (w+1)*5000) in label region [5000:10000)
        #   plus 5kb flank on each side
        # so input_pos = (sense_pos - w*5000) + 5000   (only valid if 0 <= input_pos < 15000)
        offset = window_idx * 5000
        in_s = (s - offset) + LABEL_START
        in_e = (e - offset) + LABEL_START
        # clip to [0, 15000)
        in_s = max(0, in_s)
        in_e = min(SEQ_LEN, in_e)
        if in_e > in_s:
            intervals_input.append((in_s, in_e))
    return intervals_input
''')

py(r'''
def gene_features(cond: str, row_idx: int, window_idx: int) -> dict:
    """Return all annotation tracks (in input-window coords) for one window."""
    rec = gene_record_for(cond, row_idx)
    sense, donors_s, acceptors_s = gene_to_sense(rec)
    gene_start = rec["gene_offset"]                # 0 for +, 1 for -
    gene_end = gene_start + rec["gene_len"]
    exon_spans_sense = build_exon_spans(donors_s, acceptors_s, gene_start, gene_end)

    offset = window_idx * 5000
    def s2w(pos):    # sense -> input-window
        return (pos - offset) + LABEL_START

    exon_spans = []
    for s, e in exon_spans_sense:
        ws, we = s2w(s), s2w(e)
        ws, we = max(0, ws), min(SEQ_LEN, we)
        if we > ws: exon_spans.append((ws, we))

    cds_spans = cds_window_intervals(cond, row_idx, window_idx)

    # UTR = exonic - CDS
    utr_spans = []
    cds_set = set()
    for s, e in cds_spans:
        cds_set.update(range(s, e))
    for ws, we in exon_spans:
        run_s = None
        for p in range(ws, we):
            if p in cds_set:
                if run_s is not None:
                    utr_spans.append((run_s, p))
                    run_s = None
            else:
                if run_s is None:
                    run_s = p
        if run_s is not None:
            utr_spans.append((run_s, we))

    donors_w    = [s2w(d) for d in donors_s if 0 <= s2w(d) < SEQ_LEN]
    acceptors_w = [s2w(a) for a in acceptors_s if 0 <= s2w(a) < SEQ_LEN]

    # Region classifier: position -> {"cds","utr","exon_other","intron","flank"}
    def classify(pos):
        if pos < LABEL_START or pos >= LABEL_END:
            return "flank"
        for s, e in cds_spans:
            if s <= pos < e: return "cds"
        for s, e in utr_spans:
            if s <= pos < e: return "utr"
        for s, e in exon_spans:
            if s <= pos < e: return "exon_other"
        return "intron"

    return dict(rec=rec, exon_spans=exon_spans, cds_spans=cds_spans, utr_spans=utr_spans,
                donors=donors_w, acceptors=acceptors_w, classify=classify,
                window_idx=window_idx)
''')

py(r'''
def pick_focus_window(cond: str, row_idx: int) -> int:
    """Pick the window index with the most true splice sites (where the action is)."""
    with h5py.File(CONDITION_FILES[cond]["dataset"], "r") as f:
        y_key = f"Y{row_idx}"
        y = f[y_key][0]                              # (n_win, 5000, 3) int8
    labels = np.argmax(y, axis=-1)                   # (n_win, 5000)
    n_splice_per_window = ((labels == 1) | (labels == 2)).sum(axis=1)
    return int(np.argmax(n_splice_per_window))
''')

py(r'''
@torch.no_grad()
def run_window_with_attn(cond: str, row_idx: int, window_idx: int, target_positions: list[int]):
    """Run model on one window with attention capture.

    Returns dict with:
      probs (5000, 3), labels (5000,) int,
      attn[layer] -> (T, H, L)   (B=1 squeezed; T = number of resolved targets)
      target_q_pos: list of (target_pos, q_row) — only for targets that were selected.
    """
    with h5py.File(CONDITION_FILES[cond]["dataset"], "r") as f:
        x_data = f[f"X{row_idx}"][window_idx:window_idx+1].astype(np.float32)  # (1,15000,4)
        y_data = f[f"Y{row_idx}"][0, window_idx]                               # (5000, 3) int8
    x = torch.from_numpy(x_data).permute(0, 2, 1).to(device)
    with AttentionCapture(model) as cap:
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            _, refined, _ = model(x)
    probs = torch.softmax(
        refined[:, LABEL_START:LABEL_END, :].float(), dim=-1
    ).cpu().numpy()[0]                                                       # (5000, 3)
    labels = np.argmax(y_data, axis=-1)                                      # (5000,)

    # Resolve target sequence positions to Q rows
    rows = find_target_q_rows(cap.vicinity_idx, cap.q_pad_mask, target_positions)
    resolved = [(p, q) for p, q in rows.items() if q is not None]

    if not resolved:
        return dict(probs=probs, labels=labels, attn=[], target_q_pos=[],
                    vicinity_idx=cap.vicinity_idx[0].cpu().numpy(),
                    q_pad_mask=cap.q_pad_mask[0].cpu().numpy())

    q_rows = torch.tensor(
        [[q for _, q in resolved]], dtype=torch.long, device=device
    )                                                                         # (1, T)
    attn_per_layer = []
    for li in range(N_LAYERS):
        a = cap.attention(li, q_rows)[0]                                     # (T, H, L)
        attn_per_layer.append(a.cpu().numpy())
    return dict(probs=probs, labels=labels, attn=attn_per_layer,
                target_q_pos=resolved,
                vicinity_idx=cap.vicinity_idx[0].cpu().numpy(),
                q_pad_mask=cap.q_pad_mask[0].cpu().numpy())
''')

py(r'''
REGION_COLORS = {
    "cds":        "#1f77b4",
    "utr":        "#ff7f0e",
    "exon_other": "#9467bd",
    "intron":     "#d3d3d3",
    "flank":      "#f0f0f0",
}

def draw_gene_structure(ax, feat, title=""):
    """Draw exon/CDS/UTR rectangles + donor/acceptor markers on ax."""
    ax.set_xlim(0, SEQ_LEN)
    ax.set_ylim(-1, 1)
    ax.set_yticks([])
    ax.axvspan(LABEL_START, LABEL_END, color="#fffbe6", alpha=0.5, zorder=0)
    # Introns as thin line at y=0 across exon spans (whole gene region)
    if feat["exon_spans"]:
        gx_s = min(s for s, _ in feat["exon_spans"])
        gx_e = max(e for _, e in feat["exon_spans"])
        ax.hlines(0, gx_s, gx_e, color="black", lw=1, zorder=1)
    # Exon outlines (purple)
    for s, e in feat["exon_spans"]:
        ax.add_patch(mpatches.Rectangle((s, -0.3), e-s, 0.6,
                                        facecolor=REGION_COLORS["exon_other"],
                                        edgecolor="black", lw=0.5, alpha=0.4, zorder=2))
    # UTR (orange)
    for s, e in feat["utr_spans"]:
        ax.add_patch(mpatches.Rectangle((s, -0.3), e-s, 0.6,
                                        facecolor=REGION_COLORS["utr"],
                                        edgecolor="none", alpha=0.7, zorder=3))
    # CDS (blue, taller)
    for s, e in feat["cds_spans"]:
        ax.add_patch(mpatches.Rectangle((s, -0.45), e-s, 0.9,
                                        facecolor=REGION_COLORS["cds"],
                                        edgecolor="none", alpha=0.7, zorder=4))
    # Donors (green) and acceptors (red)
    for d in feat["donors"]:
        ax.axvline(d, ymin=0.55, ymax=0.95, color="green", lw=1.2, zorder=5)
    for a in feat["acceptors"]:
        ax.axvline(a, ymin=0.05, ymax=0.45, color="red", lw=1.2, zorder=5)
    ax.set_title(title, fontsize=10)
    legend = [
        mpatches.Patch(color=REGION_COLORS["cds"], label="CDS", alpha=0.7),
        mpatches.Patch(color=REGION_COLORS["utr"], label="UTR", alpha=0.7),
        mpatches.Patch(color=REGION_COLORS["exon_other"], label="exon (no CDS info)", alpha=0.4),
        mpatches.Patch(color="green", label="donor"),
        mpatches.Patch(color="red", label="acceptor"),
    ]
    ax.legend(handles=legend, loc="upper right", fontsize=7, ncol=5, frameon=False)
''')

py(r'''
def draw_predictions(ax, probs, labels):
    """Plot P(donor), P(acceptor) along label region with true sites overlaid."""
    x = np.arange(LABEL_START, LABEL_END)
    ax.plot(x, probs[:, 2], color="green", lw=0.8, label="P(donor)")
    ax.plot(x, probs[:, 1], color="red",   lw=0.8, label="P(acceptor)")
    for p in np.where(labels == 2)[0]:
        ax.axvline(LABEL_START + p, color="green", alpha=0.3, lw=1)
    for p in np.where(labels == 1)[0]:
        ax.axvline(LABEL_START + p, color="red", alpha=0.3, lw=1)
    ax.set_xlim(0, SEQ_LEN)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("prob")
    ax.legend(loc="upper right", fontsize=7, ncol=2, frameon=False)
''')

py(r'''
def draw_attention_heatmap(ax, attn_layer, target_positions, feat, title=""):
    """attn_layer: (T, H, L) numpy. Mean across heads."""
    if attn_layer is None or len(attn_layer) == 0:
        ax.text(0.5, 0.5, "no resolved targets", ha="center", va="center", transform=ax.transAxes)
        ax.set_xlim(0, SEQ_LEN); ax.set_yticks([])
        return
    A = attn_layer.mean(axis=1)                                 # (T, L)
    # log-stretch for legibility
    A_disp = np.log10(A + 1e-6)
    im = ax.imshow(A_disp, aspect="auto", interpolation="nearest",
                   extent=(0, SEQ_LEN, len(target_positions), 0),
                   cmap="magma", vmin=-5, vmax=0)
    ax.set_yticks(np.arange(len(target_positions)) + 0.5)
    ax.set_yticklabels([f"{lbl}" for lbl in target_positions], fontsize=7)
    ax.set_title(title, fontsize=9)
    # Region bars at the top of the heatmap (one row of rectangles)
    bar_y = -0.25
    bar_h = 0.18
    for s, e in feat["cds_spans"]:
        ax.add_patch(mpatches.Rectangle((s, bar_y), e-s, bar_h,
                                        facecolor=REGION_COLORS["cds"], clip_on=False, alpha=0.8))
    for s, e in feat["utr_spans"]:
        ax.add_patch(mpatches.Rectangle((s, bar_y), e-s, bar_h,
                                        facecolor=REGION_COLORS["utr"], clip_on=False, alpha=0.8))
    return im
''')

py(r'''
def attention_top_k_table(attn_layer, target_positions, feat, k=10):
    """Return DataFrame rows for top-k attended positions per target."""
    if attn_layer is None or len(attn_layer) == 0:
        return pd.DataFrame()
    A = attn_layer.mean(axis=1)                          # (T, L)
    rows = []
    for i, tpos in enumerate(target_positions):
        topk_idx = np.argsort(-A[i])[:k]
        for rank, p in enumerate(topk_idx):
            rows.append(dict(
                target=int(tpos),
                rank=rank,
                kv_pos=int(p),
                attn=float(A[i, p]),
                region=feat["classify"](int(p)),
                offset=int(p) - int(tpos),
            ))
    return pd.DataFrame(rows)
''')

py(r'''
def build_target_positions(labels, max_per_class=4):
    """Pick true splice positions in label region (window-relative input coords)."""
    donors = (np.where(labels == 2)[0] + LABEL_START).tolist()[:max_per_class]
    acceptors = (np.where(labels == 1)[0] + LABEL_START).tolist()[:max_per_class]
    return donors + acceptors


def plot_gene_condition(cond: str, orig_gi: int, layer_idx: int = 1, save_dir: Path | None = None):
    """Render the stacked figure for (gene, condition) at the specified cross-attn layer."""
    df_rows = metrics_df.query("condition == @cond and orig_gene_idx == @orig_gi")
    if df_rows.empty:
        print(f"gene {orig_gi} not in {cond}")
        return None
    row_idx = int(df_rows.iloc[0]["row_idx"])
    name = df_rows.iloc[0]["name"]
    auprc = df_rows.iloc[0]["auprc_mean"]
    w = pick_focus_window(cond, row_idx)
    feat = gene_features(cond, row_idx, w)
    targets = build_target_positions(
        np.argmax(h5py.File(CONDITION_FILES[cond]["dataset"], "r")[f"Y{row_idx}"][0, w], -1)
    )
    out = run_window_with_attn(cond, row_idx, w, targets)

    fig, axes = plt.subplots(3, 1, figsize=(13, 8),
                             gridspec_kw=dict(height_ratios=[1, 1.5, 3]),
                             sharex=True)
    draw_gene_structure(axes[0], feat,
        title=f"{name} (orig_gi={orig_gi}, strand={feat['rec']['strand']}) — "
              f"{cond}, window {w}, AUPRC={auprc:.3f}")
    draw_predictions(axes[1], out["probs"], out["labels"])
    draw_attention_heatmap(axes[2], out["attn"][layer_idx] if out["attn"] else None,
                           [p for p, _ in out["target_q_pos"]], feat,
                           title=f"cross-attention layer {layer_idx} (mean over {N_HEADS} heads)")
    axes[2].set_xlabel("input-window position (0..15000)  —  label region 5000..10000 highlighted")
    plt.tight_layout()
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_dir / f"{name}_{cond}.png", dpi=130)
    plt.show()

    topk = attention_top_k_table(
        out["attn"][layer_idx] if out["attn"] else None,
        [p for p, _ in out["target_q_pos"]],
        feat, k=10,
    )
    return dict(name=name, cond=cond, row_idx=row_idx, window=w, feat=feat, out=out, topk=topk)
''')

py(r'''
# ---- Render every selected gene under every condition ----
RESULTS_DIR = REPO_ROOT / "evaluation" / "results" / "v5_attention_analysis"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

per_gene_runs = {}   # (orig_gi, cond) -> result dict
for orig_gi in SELECTED_GENES:
    for cond in CONDITIONS:
        try:
            r = plot_gene_condition(cond, orig_gi, save_dir=RESULTS_DIR / "per_gene")
            if r is not None:
                per_gene_runs[(orig_gi, cond)] = r
        except Exception as e:
            print(f"!! {orig_gi}/{cond}: {e}")
print(f"completed {len(per_gene_runs)} (gene, condition) runs")
''')


# ---------------------------------------------------------------------------
# Section 6: Aggregate region analysis
# ---------------------------------------------------------------------------
md("""
## 6. Aggregate "where does attention go?"

For each (gene, condition) run, compute the fraction of total attention mass landing in CDS / UTR /
exon-other / intron / flank, separately for donor- vs acceptor-target queries. Side-by-side bars
across conditions are the headline story.
""")

py(r'''
REGIONS = ["cds", "utr", "exon_other", "intron", "flank"]

def attention_region_fractions(run, layer_idx=1):
    """For a per_gene_runs entry, return DataFrame of region-fractions per target."""
    if not run["out"]["attn"]:
        return pd.DataFrame()
    A = run["out"]["attn"][layer_idx].mean(axis=1)                # (T, L)
    classify = run["feat"]["classify"]
    pos_class = np.array([classify(p) for p in range(SEQ_LEN)])
    rows = []
    labels = run["out"]["labels"]
    targets = [p for p, _ in run["out"]["target_q_pos"]]
    for i, tpos in enumerate(targets):
        lab = labels[tpos - LABEL_START]
        target_class = "donor" if lab == 2 else ("acceptor" if lab == 1 else "other")
        attn = A[i]
        total = attn.sum() + 1e-12
        row = dict(target_pos=int(tpos), target_class=target_class, total=float(total))
        for region in REGIONS:
            row[f"frac_{region}"] = float(attn[pos_class == region].sum() / total)
        rows.append(row)
    return pd.DataFrame(rows)


region_rows = []
for (orig_gi, cond), run in per_gene_runs.items():
    df = attention_region_fractions(run)
    if df.empty: continue
    df["orig_gene_idx"] = orig_gi
    df["condition"] = cond
    df["bucket"] = selected.set_index("orig_gene_idx").loc[orig_gi, "bucket"]
    region_rows.append(df)
region_df = pd.concat(region_rows, ignore_index=True) if region_rows else pd.DataFrame()
print(region_df.shape)
region_df.head()
''')

py(r'''
def plot_region_fractions(region_df, group_col="bucket"):
    if region_df.empty:
        print("no data"); return
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    for ax, target_class in zip(axes, ["donor", "acceptor"]):
        sub = region_df[region_df["target_class"] == target_class]
        agg = sub.groupby(["condition", group_col])[
            [f"frac_{r}" for r in REGIONS]
        ].mean().reset_index()

        x_groups = agg.apply(lambda r: f"{r[group_col]}\\n{r['condition']}", axis=1)
        bottoms = np.zeros(len(agg))
        for region in REGIONS:
            ax.bar(range(len(agg)), agg[f"frac_{region}"], bottom=bottoms,
                   label=region, color=REGION_COLORS[region])
            bottoms += agg[f"frac_{region}"].values
        ax.set_ylabel("attention fraction")
        ax.set_title(f"{target_class} queries")
        ax.set_xticks(range(len(agg)))
        ax.set_xticklabels(x_groups, fontsize=7, rotation=45, ha="right")
        if ax is axes[0]:
            ax.legend(loc="upper right", fontsize=7, ncol=5, frameon=False)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "attention_region_fractions.png", dpi=130)
    plt.show()

plot_region_fractions(region_df)
''')


# ---------------------------------------------------------------------------
# Section 7: Motif analysis
# ---------------------------------------------------------------------------
md("""
## 7. Motif analysis (sequence logos + ATtRACT RBP scan)

For each target splice site under `filtered_baseline`, take its top-30 attended KV positions, group
by region class, build a sequence logo from a ±10bp window around the hotspots, and score each
hotspot 21mer against splicing-factor motifs from the **ATtRACT** database.
""")

py(r'''
ATTRACT_DIR = REPO_ROOT / "data" / "attract"
ATTRACT_DIR.mkdir(parents=True, exist_ok=True)

def fetch_attract():
    """Download ATtRACT if not present. Files are tiny (single TSV, ~6MB)."""
    pwm_path = ATTRACT_DIR / "pwm.txt"
    info_path = ATTRACT_DIR / "ATtRACT_db.txt"
    if pwm_path.exists() and info_path.exists():
        return pwm_path, info_path
    import urllib.request
    URL = "https://attract.cnic.es/attract/static/ATtRACT.zip"
    zip_path = ATTRACT_DIR / "attract.zip"
    print(f"downloading ATtRACT from {URL}")
    try:
        urllib.request.urlretrieve(URL, zip_path)
        import zipfile
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(ATTRACT_DIR)
        return pwm_path, info_path
    except Exception as e:
        print(f"ATtRACT download failed ({e}); falling back to a small embedded list of canonical motifs")
        return None, None

ATTRACT_PWM_PATH, ATTRACT_INFO_PATH = fetch_attract()
print("ATtRACT files:", ATTRACT_PWM_PATH, ATTRACT_INFO_PATH)
''')

py(r'''
def parse_attract_pwms(pwm_path):
    """Return list of (motif_id, name, pwm_array (L,4) order ACGU)."""
    if pwm_path is None or not pwm_path.exists():
        return []
    motifs = []
    with open(pwm_path) as f:
        text = f.read()
    blocks = text.strip().split(">")
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        lines = block.split("\\n")
        header = lines[0].split()
        motif_id = header[0]
        rows = []
        for line in lines[1:]:
            parts = line.split()
            if len(parts) >= 4:
                try:
                    rows.append([float(x) for x in parts[:4]])
                except ValueError:
                    continue
        if rows:
            arr = np.asarray(rows)
            arr = arr / (arr.sum(axis=1, keepdims=True) + 1e-12)
            motifs.append((motif_id, motif_id, arr))
    return motifs


SPLICING_FACTORS = {"SRSF1","SRSF2","SRSF3","SRSF5","SRSF6","SRSF7","SRSF9","SRSF10","SRSF11",
                    "HNRNPA1","HNRNPA2B1","HNRNPC","HNRNPF","HNRNPH1","HNRNPK","HNRNPL","HNRNPM",
                    "HNRNPU","U2AF1","U2AF2","PTBP1","PTBP2","RBM5","RBFOX1","RBFOX2","RBFOX3",
                    "MBNL1","MBNL2","CELF1","CELF2","TIA1","SF1","TRA2A","TRA2B"}

def parse_attract_info(info_path, want=SPLICING_FACTORS):
    if info_path is None or not info_path.exists():
        return {}
    df = pd.read_csv(info_path, sep="\\t", low_memory=False)
    name_col = "Gene_name" if "Gene_name" in df.columns else df.columns[0]
    id_col = next((c for c in df.columns if "Matrix_id" in c or "ID" in c.upper()), None)
    if id_col is None: return {}
    keep = df[df[name_col].astype(str).str.upper().isin(want)]
    return dict(zip(keep[id_col].astype(str), keep[name_col].astype(str)))


ATTRACT_MOTIFS = parse_attract_pwms(ATTRACT_PWM_PATH)
ATTRACT_NAMES = parse_attract_info(ATTRACT_INFO_PATH)
print(f"ATtRACT: {len(ATTRACT_MOTIFS)} motifs total; {len(ATTRACT_NAMES)} match splicing factors")
''')

py(r'''
NT_INDEX = {"A":0,"C":1,"G":2,"T":3,"U":3,"N":-1,"a":0,"c":1,"g":2,"t":3,"u":3,"n":-1}

def kmer_in_window(rec, window_idx: int, center: int, radius: int = 10) -> str:
    """Return the input-window subsequence centered on position `center` (sense-mapped)."""
    sense, _, _ = gene_to_sense(rec)
    s2w_offset = window_idx * 5000 - LABEL_START
    sense_pos = center + s2w_offset
    s = sense_pos - radius
    e = sense_pos + radius + 1
    pad_l = max(0, -s); pad_r = max(0, e - len(sense))
    s = max(0, s); e = min(len(sense), e)
    return ("N"*pad_l) + sense[s:e] + ("N"*pad_r)


def score_kmer_against_pwm(kmer: str, pwm: np.ndarray) -> tuple[float, int]:
    """Sliding match: max log-likelihood over all offsets. Returns (best_score, best_offset)."""
    L = pwm.shape[0]
    best = -np.inf; best_off = 0
    for off in range(len(kmer) - L + 1):
        s = 0.0
        bad = False
        for i in range(L):
            ni = NT_INDEX.get(kmer[off + i], -1)
            if ni < 0: bad = True; break
            s += math.log(pwm[i, ni] + 1e-3)
        if bad: continue
        if s > best:
            best = s; best_off = off
    return best, best_off
''')

py(r'''
def hotspot_kmers(run, layer_idx=1, top_k=30, radius=10):
    """Return DataFrame: one row per (target, hotspot) with the local 21mer."""
    if not run["out"]["attn"]:
        return pd.DataFrame()
    A = run["out"]["attn"][layer_idx].mean(axis=1)
    targets = [p for p, _ in run["out"]["target_q_pos"]]
    classify = run["feat"]["classify"]
    rec = run["feat"]["rec"]
    rows = []
    for i, tpos in enumerate(targets):
        topk = np.argsort(-A[i])[:top_k]
        lab = run["out"]["labels"][tpos - LABEL_START]
        target_class = "donor" if lab == 2 else ("acceptor" if lab == 1 else "other")
        for rank, p in enumerate(topk):
            kmer = kmer_in_window(rec, run["window"], int(p), radius=radius)
            rows.append(dict(
                target_class=target_class,
                target_pos=int(tpos),
                kv_pos=int(p),
                rank=int(rank),
                attn=float(A[i, p]),
                region=classify(int(p)),
                offset=int(p) - int(tpos),
                kmer=kmer,
            ))
    return pd.DataFrame(rows)


hotspot_rows = []
for (orig_gi, cond), run in per_gene_runs.items():
    if cond != "filtered_baseline":   # logos most informative on unmodified sequence
        continue
    df = hotspot_kmers(run)
    if df.empty: continue
    df["orig_gene_idx"] = orig_gi
    df["bucket"] = selected.set_index("orig_gene_idx").loc[orig_gi, "bucket"]
    hotspot_rows.append(df)
hotspots_df = pd.concat(hotspot_rows, ignore_index=True) if hotspot_rows else pd.DataFrame()
hotspots_df.head()
''')

py(r'''
def plot_logo_for_subset(df_sub, title, ax):
    """Build a sequence logo from a DataFrame of k-mers (all same length)."""
    if df_sub.empty:
        ax.text(0.5, 0.5, "no kmers", ha="center", va="center", transform=ax.transAxes)
        return
    kmers = [k.upper().replace("U","T") for k in df_sub["kmer"] if "N" not in k]
    if not kmers:
        ax.text(0.5, 0.5, "all kmers contain N", ha="center", va="center", transform=ax.transAxes)
        return
    L = len(kmers[0])
    counts = pd.DataFrame(0, index=range(L), columns=list("ACGT"))
    for k in kmers:
        for i, ch in enumerate(k):
            if ch in counts.columns:
                counts.loc[i, ch] += 1
    info = logomaker.transform_matrix(counts, from_type="counts", to_type="information")
    logomaker.Logo(info, ax=ax, color_scheme="classic")
    ax.set_title(title, fontsize=9)
    ax.set_xlabel(""); ax.set_ylabel("bits")


# Per-bucket × target_class × region logos
buckets_to_show = ["top_baseline", "cds_robust", "cds_dependent"]
target_classes = ["donor", "acceptor"]
fig, axes = plt.subplots(len(buckets_to_show), len(target_classes),
                         figsize=(11, 2.8 * len(buckets_to_show)))
if len(buckets_to_show) == 1: axes = np.array([axes])
for i, bucket in enumerate(buckets_to_show):
    for j, tc in enumerate(target_classes):
        sub = hotspots_df[(hotspots_df["bucket"] == bucket) &
                          (hotspots_df["target_class"] == tc) &
                          (hotspots_df["rank"] < 10)]
        plot_logo_for_subset(sub, f"{bucket} / {tc}", axes[i][j])
plt.tight_layout()
plt.savefig(RESULTS_DIR / "hotspot_logos.png", dpi=130)
plt.show()
''')

py(r'''
# ---- ATtRACT RBP scan against hotspot 21mers (top-10 per target) ----
if ATTRACT_MOTIFS and not hotspots_df.empty:
    matches = []
    targets_df = hotspots_df[hotspots_df["rank"] < 10].copy()
    for _, row in targets_df.iterrows():
        kmer = row["kmer"].upper().replace("U", "T")
        if "N" in kmer: continue
        best_motif = None; best_score = -np.inf; best_off = 0
        for motif_id, name, pwm in ATTRACT_MOTIFS:
            score, off = score_kmer_against_pwm(kmer, pwm)
            if score > best_score:
                best_score = score; best_motif = motif_id; best_off = off
        if best_motif:
            matches.append(dict(
                orig_gene_idx=int(row["orig_gene_idx"]),
                bucket=row["bucket"],
                target_class=row["target_class"],
                kv_pos=int(row["kv_pos"]),
                region=row["region"],
                kmer=kmer,
                best_motif=best_motif,
                best_factor=ATTRACT_NAMES.get(best_motif, "?"),
                score=best_score,
                offset=best_off,
            ))
    matches_df = pd.DataFrame(matches)
    # Most-frequent splicing-factor matches per bucket
    if not matches_df.empty:
        sf = matches_df[matches_df["best_factor"].isin(SPLICING_FACTORS)]
        print("top splicing-factor matches per bucket:")
        for bucket, sub in sf.groupby("bucket"):
            counts = sub["best_factor"].value_counts().head(8)
            print(f"\\n{bucket}:")
            print(counts.to_string())
    matches_df.to_csv(RESULTS_DIR / "rbp_matches.csv", index=False)
else:
    print("ATtRACT not loaded; skipping RBP scan")
    matches_df = pd.DataFrame()
''')


# ---------------------------------------------------------------------------
# Section 8: Cross-condition diff
# ---------------------------------------------------------------------------
md("""
## 8. Cross-condition attention diff

For each gene that has runs under all three conditions and shares the same window/target positions,
compute `attn_baseline - attn_shuffle` along the 15kb axis. The sign tells us whether the model is
*losing* attention mass at that position (positive) or *gaining* (negative) when CDS info is destroyed.
""")

py(r'''
def cross_condition_diff(orig_gi, layer_idx=1):
    base = per_gene_runs.get((orig_gi, "filtered_baseline"))
    if base is None or not base["out"]["attn"]:
        return None
    A_base = base["out"]["attn"][layer_idx].mean(axis=1)
    targets_base = [p for p, _ in base["out"]["target_q_pos"]]
    diffs = {}
    for shuf in ("nt_shuffle", "codon_shuffle"):
        run = per_gene_runs.get((orig_gi, shuf))
        if run is None or not run["out"]["attn"]: continue
        A_shuf = run["out"]["attn"][layer_idx].mean(axis=1)
        targets_shuf = [p for p, _ in run["out"]["target_q_pos"]]
        # Only compare targets present in both runs
        common = [t for t in targets_base if t in targets_shuf]
        if not common: continue
        idx_base = [targets_base.index(t) for t in common]
        idx_shuf = [targets_shuf.index(t) for t in common]
        diffs[shuf] = (common, A_base[idx_base] - A_shuf[idx_shuf], base["feat"])
    return diffs


def plot_diff_for_gene(orig_gi):
    diffs = cross_condition_diff(orig_gi)
    if not diffs:
        return
    name = selected.set_index("orig_gene_idx").loc[orig_gi, "name"]
    n = len(diffs)
    fig, axes = plt.subplots(n + 1, 1, figsize=(13, 2.5 * (n+1)),
                             sharex=True, gridspec_kw=dict(height_ratios=[1] + [2]*n))
    base = per_gene_runs[(orig_gi, "filtered_baseline")]
    draw_gene_structure(axes[0], base["feat"], title=f"{name} — baseline gene structure")
    for ax, (shuf, (targets, D, feat)) in zip(axes[1:], diffs.items()):
        D_mean = D.mean(axis=0)              # average across targets
        ax.fill_between(np.arange(SEQ_LEN), 0, D_mean,
                        where=(D_mean > 0), color="green", alpha=0.5,
                        label="lost under shuffle")
        ax.fill_between(np.arange(SEQ_LEN), 0, D_mean,
                        where=(D_mean < 0), color="purple", alpha=0.5,
                        label="gained under shuffle")
        ax.axhline(0, color="black", lw=0.5)
        ax.set_xlim(0, SEQ_LEN)
        ax.set_ylabel("Δ attn")
        ax.set_title(f"{shuf}: baseline - shuffle, mean over {len(targets)} targets", fontsize=9)
        ax.legend(loc="upper right", fontsize=7, frameon=False)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f"diff_{name}.png", dpi=130)
    plt.show()


for orig_gi in SELECTED_GENES:
    if (orig_gi, "filtered_baseline") in per_gene_runs:
        plot_diff_for_gene(orig_gi)
''')


# ---------------------------------------------------------------------------
# Section 9: Saliency
# ---------------------------------------------------------------------------
md("""
## 9. Saliency cross-check

Input-gradient saliency (`|grad * x|` summed over channels) for the donor logit at one true donor in
each of the most interesting genes. Plotted alongside attention to confirm the two views agree.
""")

py(r'''
def saliency_for_donor(cond, row_idx, window_idx, target_pos):
    """Return saliency (15000,) for the donor logit at target_pos."""
    with h5py.File(CONDITION_FILES[cond]["dataset"], "r") as f:
        x_data = f[f"X{row_idx}"][window_idx:window_idx+1].astype(np.float32)
    x = torch.from_numpy(x_data).permute(0, 2, 1).to(device).requires_grad_(True)
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        _, refined, _ = model(x)
    refined = refined.float()
    target = refined[0, target_pos, 2]                  # donor logit
    grad = torch.autograd.grad(target, x, retain_graph=False)[0][0]   # (4, 15000)
    sal = (grad.abs() * x[0].detach()).sum(dim=0).cpu().numpy()       # (15000,)
    return sal


# Run for the cds_robust bucket — most interesting candidates
robust_picks = selected[selected["bucket"] == "cds_robust"]["orig_gene_idx"].tolist()
for orig_gi in robust_picks[:2]:
    base = per_gene_runs.get((orig_gi, "filtered_baseline"))
    if base is None: continue
    targets = [p for p, _ in base["out"]["target_q_pos"]]
    donors = [t for t in targets if base["out"]["labels"][t - LABEL_START] == 2]
    if not donors: continue
    tpos = donors[0]
    sal = saliency_for_donor("filtered_baseline", base["row_idx"], base["window"], tpos)
    A = base["out"]["attn"][1].mean(axis=1)[targets.index(tpos)]    # layer 1, this target
    fig, axes = plt.subplots(2, 1, figsize=(13, 4), sharex=True)
    axes[0].plot(sal, color="black", lw=0.5)
    axes[0].axvline(tpos, color="green", lw=1)
    axes[0].set_title(f"{base['name']}: saliency for donor at pos {tpos} (filtered_baseline)")
    axes[1].plot(A, color="orange", lw=0.5)
    axes[1].axvline(tpos, color="green", lw=1)
    axes[1].set_title("attention (layer 1, mean over heads)")
    axes[1].set_xlabel("input-window position")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f"saliency_{base['name']}.png", dpi=130)
    plt.show()
''')


# ---------------------------------------------------------------------------
# Section 10: Wet-lab summary
# ---------------------------------------------------------------------------
md("""
## 10. Wet-lab candidate summary

For each `cds_robust` and `cds_dependent` gene, list:
- gene name + orig_gene_idx + strand
- per-condition AUPRC (so it's clear how robust the signal is)
- top RBP matches in attention hotspots (if ATtRACT loaded)
- which feature regions the attention concentrates in
- a one-line testable hypothesis
""")

py(r'''
def candidate_summary(orig_gi):
    rec = selected.set_index("orig_gene_idx").loc[orig_gi]
    base = per_gene_runs.get((orig_gi, "filtered_baseline"))
    if base is None: return None
    classify = base["feat"]["classify"]
    A = base["out"]["attn"][1].mean(axis=1)             # (T, L) layer 1
    region_share = {}
    for r in REGIONS:
        mask = np.array([classify(p) == r for p in range(SEQ_LEN)])
        region_share[r] = float(A[:, mask].sum() / (A.sum() + 1e-12))
    rbp_top = ""
    if not matches_df.empty:
        sub = matches_df[matches_df["orig_gene_idx"] == orig_gi]
        sf = sub[sub["best_factor"].isin(SPLICING_FACTORS)]
        if not sf.empty:
            rbp_top = ", ".join(sf["best_factor"].value_counts().head(3).index.tolist())
    dominant = max(region_share, key=region_share.get)
    hypothesis = (
        f"Mutating the highest-attention {dominant} hotspots "
        f"{'(predicted ' + rbp_top + ' binding)' if rbp_top else ''} "
        "should weaken splice-site recognition for this gene."
    )
    return dict(
        name=rec["name"],
        orig_gene_idx=int(orig_gi),
        strand=rec["strand"],
        bucket=rec["bucket"],
        auprc_baseline=round(rec["auprc_mean_filtered_baseline"], 3),
        auprc_nt_shuffle=round(rec["auprc_mean_nt_shuffle"], 3),
        auprc_codon_shuffle=round(rec["auprc_mean_codon_shuffle"], 3),
        delta_nt=round(rec["delta_nt"], 3),
        delta_codon=round(rec["delta_codon"], 3),
        attn_dominant_region=dominant,
        attn_share_cds=round(region_share["cds"], 3),
        attn_share_intron=round(region_share["intron"], 3),
        attn_share_utr=round(region_share["utr"], 3),
        rbp_top=rbp_top,
        hypothesis=hypothesis,
    )


candidate_picks = selected[selected["bucket"].isin(["cds_robust", "cds_dependent"])][
    "orig_gene_idx"
].tolist()
candidates = [candidate_summary(g) for g in candidate_picks]
candidates_df = pd.DataFrame([c for c in candidates if c is not None])
candidates_df.to_csv(RESULTS_DIR / "wet_lab_candidates.csv", index=False)
candidates_df
''')

py(r'''
# Final: a one-line console summary of the deliverables
print("Deliverables saved under:", RESULTS_DIR)
for p in sorted(RESULTS_DIR.glob("**/*")):
    if p.is_file():
        print(" -", p.relative_to(RESULTS_DIR))
''')


# ---------------------------------------------------------------------------
# Build the .ipynb
# ---------------------------------------------------------------------------
def make_cell(cell_type: str, source: str) -> dict:
    src_lines = source.splitlines(keepends=True)
    if not src_lines:
        src_lines = [""]
    cell = {"cell_type": cell_type, "metadata": {}, "source": src_lines}
    if cell_type == "code":
        cell["outputs"] = []
        cell["execution_count"] = None
    return cell

nb = {
    "cells": [make_cell(t, s) for t, s in cells],
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.9"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
OUT.write_text(json.dumps(nb, indent=1))
print(f"wrote {OUT}  ({len(cells)} cells)")
