"""
In-silico context-masking experiment (experiment 2b).

For each radius R (per-side from the 15kb window center at position 7500),
zero out all one-hot channels outside [7500-R : 7500+R] before running the
model. Predictions over the label region [5000:10000] are saved as npz.

This measures how much of the provided context each model actually uses.

Usage:

    # SpliceMamba (single checkpoint — use best.pt for speed, ensemble optional)
    python evaluation/context_masking.py --model splicemamba --radius 500 \
        --checkpoint checkpoints/best.pt

    # SpliceMamba ensemble (slower but matches the cached baseline)
    python evaluation/context_masking.py --model splicemamba --radius 500 \
        --checkpoint dummy \
        --checkpoints checkpoints-ensemble/model_1/best.pt \
                      checkpoints-ensemble/model_2/best.pt \
                      checkpoints-ensemble/model_3/best.pt \
                      checkpoints-ensemble/model_4/best.pt \
                      checkpoints-ensemble/model_5/best.pt

    # SpliceAI (run via the spliceai_env python)
    /mnt/lareaulab/mparsa/miniconda3/envs/spliceai_env/bin/python \
        evaluation/context_masking.py --model spliceai --radius 500

Output: evaluation/results/behavior/context_masking_{model}_R{radius}.npz
with key "probs", shape (total_windows, 5000, 3), float32.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import h5py
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_EVAL_DIR = _REPO_ROOT / "evaluation"
if str(_EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(_EVAL_DIR))

DATASET_H5 = _REPO_ROOT / "dataset_test_0.h5"
OUT_DIR = _EVAL_DIR / "results" / "behavior"

WINDOW_LEN = 15000
WINDOW_CENTER = 7500
LABEL_START = 5000
LABEL_END = 10000


def mask_window(x: np.ndarray, radius: int) -> np.ndarray:
    """Zero out one-hot channels outside [center-radius : center+radius].

    x: (..., 15000, 4) one-hot encoded input. Returns a copy with outer
    regions set to 0."""
    x = x.copy()
    lo = max(0, WINDOW_CENTER - radius)
    hi = min(WINDOW_LEN, WINDOW_CENTER + radius)
    if lo > 0:
        x[..., :lo, :] = 0
    if hi < WINDOW_LEN:
        x[..., hi:, :] = 0
    return x


# ---------------------------------------------------------------------------
# SpliceMamba inference with masking
# ---------------------------------------------------------------------------

def run_splicemamba(radius: int, checkpoint: str, checkpoints: list[str] | None,
                    batch_size: int = 8) -> np.ndarray:
    import torch  # local import so SpliceAI env doesn't need torch
    from model import SpliceMamba  # noqa: F401 — imported via evaluate helper

    from evaluate import EVAL_CONFIG, load_model
    cfg = dict(EVAL_CONFIG)
    cfg["batch_size"] = batch_size

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[splicemamba] device={device}, batch_size={batch_size}, radius={radius}",
          file=sys.stderr)

    if checkpoints:
        ckpt_list = list(checkpoints)
    else:
        ckpt_list = [checkpoint]

    accum = None
    t0 = time.time()
    for ci, ckpt_path in enumerate(ckpt_list):
        print(f"[splicemamba] loading model {ci + 1}/{len(ckpt_list)}: {ckpt_path}",
              file=sys.stderr)
        model = load_model(ckpt_path, cfg, device)
        model.eval()

        model_probs = []
        with h5py.File(DATASET_H5, "r") as f:
            x_keys = sorted([k for k in f.keys() if k.startswith("X")],
                            key=lambda k: int(k[1:]))
            for x_key in x_keys:
                x_data = f[x_key][:]  # (N, 15000, 4) int8
                n_win = x_data.shape[0]
                shard_probs = []
                for start in range(0, n_win, batch_size):
                    end = min(start + batch_size, n_win)
                    batch_np = x_data[start:end].astype(np.float32)
                    batch_np = mask_window(batch_np, radius)
                    batch = torch.from_numpy(batch_np).permute(0, 2, 1).to(device)
                    with torch.no_grad(), torch.autocast(
                        device_type="cuda", dtype=torch.bfloat16):
                        _, refined_logits, _ = model(batch)
                    label_logits = refined_logits[:, LABEL_START:LABEL_END, :].float()
                    probs = torch.softmax(label_logits, dim=-1).cpu().numpy()
                    shard_probs.append(probs)
                model_probs.append(np.concatenate(shard_probs, axis=0))
                print(f"[splicemamba]   {x_key}: {n_win} windows  "
                      f"(elapsed {time.time() - t0:.0f}s)", file=sys.stderr)

        probs_full = np.concatenate(model_probs, axis=0)
        if accum is None:
            accum = probs_full
        else:
            accum += probs_full
        del model
        torch.cuda.empty_cache()

    accum /= len(ckpt_list)
    print(f"[splicemamba] done in {time.time() - t0:.0f}s, shape={accum.shape}",
          file=sys.stderr)
    return accum.astype(np.float32)


# ---------------------------------------------------------------------------
# SpliceAI inference with masking
# ---------------------------------------------------------------------------

def run_spliceai(radius: int, batch_size: int = 6) -> np.ndarray:
    from pkg_resources import resource_filename
    from keras.models import load_model as keras_load

    print(f"[spliceai] radius={radius}, loading 5-model ensemble", file=sys.stderr)
    models = []
    for i in range(1, 6):
        path = resource_filename("spliceai", f"models/spliceai{i}.h5")
        print(f"[spliceai]   loading {path}", file=sys.stderr)
        models.append(keras_load(path, compile=False))

    all_probs = []
    t0 = time.time()
    with h5py.File(DATASET_H5, "r") as f:
        x_keys = sorted([k for k in f.keys() if k.startswith("X")],
                        key=lambda k: int(k[1:]))
        for x_key in x_keys:
            x_data = f[x_key][:].astype(np.float32)  # (N, 15000, 4)
            x_masked = mask_window(x_data, radius)
            n_win = x_masked.shape[0]
            preds = np.zeros((n_win, 5000, 3), dtype=np.float32)
            for m in models:
                preds += m.predict(x_masked, batch_size=batch_size, verbose=0)
            preds /= len(models)
            all_probs.append(preds)
            print(f"[spliceai]   {x_key}: {n_win} windows  "
                  f"(elapsed {time.time() - t0:.0f}s)", file=sys.stderr)
    return np.concatenate(all_probs, axis=0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["splicemamba", "spliceai"], required=True)
    ap.add_argument("--radius", type=int, required=True,
                    help="Per-side radius from window center (7500); "
                         "keep [center-R : center+R], zero the rest.")
    ap.add_argument("--checkpoint", default=None,
                    help="SpliceMamba single checkpoint (ignored if --checkpoints given)")
    ap.add_argument("--checkpoints", nargs="+", default=None,
                    help="SpliceMamba ensemble checkpoints")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.model == "splicemamba":
        probs = run_splicemamba(args.radius, args.checkpoint, args.checkpoints,
                                batch_size=args.batch_size)
    else:
        probs = run_spliceai(args.radius, batch_size=args.batch_size)

    out_path = out_dir / f"context_masking_{args.model}_R{args.radius}.npz"
    np.savez_compressed(out_path, probs=probs)
    print(f"[done] wrote {out_path}  shape={probs.shape}", file=sys.stderr)


if __name__ == "__main__":
    main()
