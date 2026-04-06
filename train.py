"""
SpliceMamba training script.

This is the ML Scripts checkpoint. For more details, see the repository.
I hope this works as project checkpoint. 
Please note, I didn't use a notebook because training big ML models takes a long time somethimes.
And Jupyter notebook crashed or doesn't show the results properly. 
So I used a python file and called in in tmux so that if I lose access to the server, I can still continue training.

Usage:
    python train.py [--resume CHECKPOINT_PATH]
"""

from __future__ import annotations


import argparse
import math
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler
from sklearn.metrics import average_precision_score
from tqdm import tqdm
import wandb

from dataset import build_train_loader
from model import SpliceMamba
from losses import FocalLoss

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CONFIG = dict(
    # Paths
    dataset_path="dataset_train_all.h5",
    datafile_path="datafile_train_all.h5",
    checkpoint_dir="checkpoints",

    # Model
    d_model=256,
    n_mamba_layers=6,
    d_state=16,
    expand=1,
    d_conv=4,
    n_attn_layers=2,
    n_heads=8,
    window_radius=200,
    dropout=0.1,
    n_classes=3,
    max_len=15000,

    # Loss
    focal_gamma=2.0,
    focal_alpha=[0.01, 1.0, 1.0],
    lambda_coarse=0.3,
    lambda_refined=1.0,
    label_start=5000,
    label_end=10000,

    # Optimizer
    lr=3e-4,
    weight_decay=0.05,
    betas=(0.9, 0.95),
    eps=1e-8,
    max_grad_norm=1.0,

    # Schedule
    warmup_steps=2000,
    min_lr=1e-5,

    # Gate temperature
    tau_initial=5.0,
    tau_final=1.0,
    tau_anneal_epochs=10,

    # Batch
    micro_batch_size=8*4,
    grad_accum_steps=4,
    effective_batch_size=32*4,

    # Training
    max_epochs=40,
    early_stopping_patience=7,
    val_fraction=0.1,
    seed=42,

    # Data loading
    num_workers=4,
)


def get_git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_gate_temperature(epoch: int, cfg: dict) -> float:
    return max(
        cfg["tau_final"],
        cfg["tau_initial"] - (cfg["tau_initial"] - cfg["tau_final"])
        / cfg["tau_anneal_epochs"] * epoch,
    )


# ---------------------------------------------------------------------------
# Learning rate scheduler (warmup + cosine decay)
# ---------------------------------------------------------------------------

class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_steps: int, total_steps: int,
                 max_lr: float, min_lr: float):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.step_count = 0

    def step(self):
        self.step_count += 1
        lr = self._get_lr()
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr

    def _get_lr(self) -> float:
        if self.step_count <= self.warmup_steps:
            return self.max_lr * self.step_count / max(1, self.warmup_steps)
        # Cosine decay
        progress = (self.step_count - self.warmup_steps) / max(
            1, self.total_steps - self.warmup_steps
        )
        progress = min(progress, 1.0)
        return self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (
            1 + math.cos(math.pi * progress)
        )

    def get_lr(self) -> float:
        return self._get_lr()

    def state_dict(self):
        return {"step_count": self.step_count}

    def load_state_dict(self, d):
        self.step_count = d["step_count"]


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

@torch.no_grad()
def validate(model, val_loader, criterion, cfg, device):
    model.eval()
    total_loss = 0.0
    all_probs = []
    all_labels = []
    n_batches = 0

    for x, y in tqdm(val_loader, desc="Validating", leave=False):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            coarse_logits, refined_logits, _ = model(x, tau=1.0)

        # Slice label region
        refined_label = refined_logits[:, cfg["label_start"]:cfg["label_end"], :]
        coarse_label = coarse_logits[:, cfg["label_start"]:cfg["label_end"], :]

        # Loss in fp32
        refined_flat = refined_label.reshape(-1, cfg["n_classes"]).float()
        coarse_flat = coarse_label.reshape(-1, cfg["n_classes"]).float()
        y_flat = y.reshape(-1)

        loss_refined = criterion(refined_flat, y_flat)
        loss_coarse = criterion(coarse_flat, y_flat)
        loss = cfg["lambda_refined"] * loss_refined + cfg["lambda_coarse"] * loss_coarse

        total_loss += loss.item()
        n_batches += 1

        # Collect predictions for AUPRC
        probs = torch.softmax(refined_flat.detach(), dim=-1).cpu().numpy()
        labels = y_flat.detach().cpu().numpy()
        all_probs.append(probs)
        all_labels.append(labels)

    avg_loss = total_loss / max(n_batches, 1)

    # Compute AUPRC
    all_probs = np.concatenate(all_probs, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    # Acceptor AUPRC: class 1 vs rest
    acc_true = (all_labels == 1).astype(np.int32)
    acc_score = all_probs[:, 1]
    auprc_acceptor = average_precision_score(acc_true, acc_score) if acc_true.sum() > 0 else 0.0

    # Donor AUPRC: class 2 vs rest
    donor_true = (all_labels == 2).astype(np.int32)
    donor_score = all_probs[:, 2]
    auprc_donor = average_precision_score(donor_true, donor_score) if donor_true.sum() > 0 else 0.0

    auprc_mean = (auprc_donor + auprc_acceptor) / 2.0

    # Top-k accuracy
    topk_results = compute_topk_accuracy(all_probs, all_labels)

    model.train()
    return {
        "loss": avg_loss,
        "auprc_donor": auprc_donor,
        "auprc_acceptor": auprc_acceptor,
        "auprc_mean": auprc_mean,
        **topk_results,
    }


def compute_topk_accuracy(probs: np.ndarray, labels: np.ndarray) -> dict:
    """Compute top-k accuracy at k = {0.5, 1, 2, 4} × true count."""
    results = {}
    for cls_name, cls_idx in [("acceptor", 1), ("donor", 2)]:
        true_mask = labels == cls_idx
        n_true = true_mask.sum()
        if n_true == 0:
            for k in [0.5, 1, 2, 4]:
                results[f"topk_{cls_name}_k{k}"] = 0.0
            continue

        scores = probs[:, cls_idx]
        for k in [0.5, 1, 2, 4]:
            n_select = max(1, int(k * n_true))
            top_indices = np.argpartition(-scores, n_select)[:n_select]
            n_found = true_mask[top_indices].sum()
            results[f"topk_{cls_name}_k{k}"] = float(n_found) / float(n_true)

    return results


# ---------------------------------------------------------------------------
# Training step monitoring helpers
# ---------------------------------------------------------------------------

def compute_gate_and_attn_stats(
    coarse_logits: torch.Tensor,
    refined_logits: torch.Tensor,
    labels: torch.Tensor,
    encoder_output: torch.Tensor,
    model: nn.Module,
    tau: float,
    cfg: dict,
) -> dict:
    """Compute gate and attention monitoring stats for logging."""
    with torch.no_grad():
        # Slice to label region
        coarse_label = coarse_logits[:, cfg["label_start"]:cfg["label_end"], :]
        enc_label = encoder_output[:, cfg["label_start"]:cfg["label_end"], :]

        # Gate values
        splice_logits = torch.max(coarse_label[:, :, 1], coarse_label[:, :, 2])
        gate = torch.sigmoid(splice_logits / tau)  # (B, 5000)

        # Masks
        labels_flat = labels.reshape(-1)
        gate_flat = gate.reshape(-1)
        splice_mask = labels_flat > 0
        neither_mask = labels_flat == 0

        stats = {}
        if splice_mask.any():
            stats["gate_mean_at_splice_sites"] = gate_flat[splice_mask].mean().item()
        else:
            stats["gate_mean_at_splice_sites"] = 0.0

        if neither_mask.any():
            stats["gate_mean_at_neither"] = gate_flat[neither_mask].mean().item()
        else:
            stats["gate_mean_at_neither"] = 0.0

        # Attention output norms — approximate from difference between
        # encoder output and refined input. For exact tracking we'd need
        # intermediate outputs, but this is a reasonable proxy.
        enc_flat = enc_label.reshape(-1, enc_label.size(-1))
        if splice_mask.any():
            stats["attn_output_norm_at_splice"] = (
                enc_flat[splice_mask].norm(dim=-1).mean().item()
            )
        else:
            stats["attn_output_norm_at_splice"] = 0.0

        if neither_mask.any():
            # Sample to avoid computing norm over millions of positions
            n_sample = min(10000, neither_mask.sum().item())
            idx = torch.where(neither_mask)[0][:n_sample]
            stats["attn_output_norm_at_neither"] = (
                enc_flat[idx].norm(dim=-1).mean().item()
            )
        else:
            stats["attn_output_norm_at_neither"] = 0.0

    return stats


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(cfg: dict, resume_path: str | None = None):
    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # W&B init
    wandb.init(
        project="splice-mamba",
        config={**cfg, "git_hash": get_git_hash()},
    )

    # Data
    print("Building data loaders...")
    train_loader, val_loader = build_train_loader(
        cfg["dataset_path"],
        cfg["datafile_path"],
        batch_size=cfg["micro_batch_size"],
        num_workers=cfg["num_workers"],
        val_fraction=cfg["val_fraction"],
        seed=cfg["seed"],
    )

    # Model
    model = SpliceMamba(
        d_model=cfg["d_model"],
        n_mamba_layers=cfg["n_mamba_layers"],
        d_state=cfg["d_state"],
        expand=cfg["expand"],
        d_conv=cfg["d_conv"],
        n_attn_layers=cfg["n_attn_layers"],
        n_heads=cfg["n_heads"],
        window_radius=cfg["window_radius"],
        dropout=cfg["dropout"],
        n_classes=cfg["n_classes"],
        max_len=cfg["max_len"],
    ).to(device)
    print(f"Model parameters: {model.count_parameters():,}")
    wandb.log({"model/parameters": model.count_parameters()})

    # Loss
    criterion = FocalLoss(
        gamma=cfg["focal_gamma"],
        alpha=cfg["focal_alpha"],
    ).to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
        betas=cfg["betas"],
        eps=cfg["eps"],
    )

    # Scheduler
    steps_per_epoch = len(train_loader) // cfg["grad_accum_steps"]
    total_steps = steps_per_epoch * cfg["max_epochs"]
    scheduler = WarmupCosineScheduler(
        optimizer, cfg["warmup_steps"], total_steps, cfg["lr"], cfg["min_lr"]
    )

    # GradScaler for mixed precision (bf16 doesn't need scaling but we keep
    # the pattern for clean gradient accumulation)
    scaler = GradScaler(enabled=False)  # bf16 doesn't need loss scaling

    # Checkpoint directory
    ckpt_dir = Path(cfg["checkpoint_dir"])
    ckpt_dir.mkdir(exist_ok=True)

    # Resume
    start_epoch = 0
    best_auprc = 0.0
    patience_counter = 0

    if resume_path and os.path.exists(resume_path):
        print(f"Resuming from {resume_path}")
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_auprc = ckpt.get("best_auprc", 0.0)
        patience_counter = ckpt.get("patience_counter", 0)
        print(f"Resumed at epoch {start_epoch}, best AUPRC: {best_auprc:.4f}")

    # -----------------------------------------------------------------------
    # Training loop
    # -----------------------------------------------------------------------

    global_step = scheduler.step_count
    model.train()

    for epoch in range(start_epoch, cfg["max_epochs"]):
        tau = get_gate_temperature(epoch, cfg)
        epoch_loss = 0.0
        epoch_steps = 0
        t0 = time.time()

        optimizer.zero_grad()

        pbar = tqdm(
            enumerate(train_loader),
            total=len(train_loader),
            desc=f"Epoch {epoch}/{cfg['max_epochs']-1}",
        )
        for batch_idx, (x, y) in pbar:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            # Forward pass in bf16
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                coarse_logits, refined_logits, encoder_output = model(x, tau=tau)

            # Slice to label region and compute loss in fp32
            coarse_label = coarse_logits[:, cfg["label_start"]:cfg["label_end"], :]
            refined_label = refined_logits[:, cfg["label_start"]:cfg["label_end"], :]

            coarse_flat = coarse_label.reshape(-1, cfg["n_classes"]).float()
            refined_flat = refined_label.reshape(-1, cfg["n_classes"]).float()
            y_flat = y.reshape(-1)

            loss_coarse = criterion(coarse_flat, y_flat)
            loss_refined = criterion(refined_flat, y_flat)
            loss = (
                cfg["lambda_refined"] * loss_refined
                + cfg["lambda_coarse"] * loss_coarse
            )
            loss = loss / cfg["grad_accum_steps"]

            # Backward
            loss.backward()

            epoch_loss += loss.item() * cfg["grad_accum_steps"]
            epoch_steps += 1

            # Update progress bar
            pbar.set_postfix(
                loss=f"{loss.item() * cfg['grad_accum_steps']:.4f}",
                lr=f"{scheduler.get_lr():.2e}",
                tau=f"{tau:.1f}",
            )

            # Optimizer step every grad_accum_steps
            if (batch_idx + 1) % cfg["grad_accum_steps"] == 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), cfg["max_grad_norm"]
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                # Log per optimizer step
                current_lr = scheduler.get_lr()
                step_log = {
                    "train/loss_total": loss.item() * cfg["grad_accum_steps"],
                    "train/loss_coarse": loss_coarse.item(),
                    "train/loss_refined": loss_refined.item(),
                    "train/learning_rate": current_lr,
                    "train/gate_temperature": tau,
                    "train/global_step": global_step,
                    "epoch": epoch,
                }

                # Gate and attention stats (every 50 steps to reduce overhead)
                if global_step % 50 == 0:
                    gate_stats = compute_gate_and_attn_stats(
                        coarse_logits, refined_logits, y,
                        encoder_output, model, tau, cfg,
                    )
                    step_log.update({
                        f"train/{k}": v for k, v in gate_stats.items()
                    })

                wandb.log(step_log, step=global_step)

        pbar.close()

        # End of epoch
        epoch_time = time.time() - t0
        avg_train_loss = epoch_loss / max(epoch_steps, 1)
        print(f"Epoch {epoch}/{cfg['max_epochs']-1} | "
              f"train_loss: {avg_train_loss:.4f} | "
              f"tau: {tau:.2f} | "
              f"time: {epoch_time/60:.1f}min")

        # Validation
        print("Running validation...")
        val_metrics = validate(model, val_loader, criterion, cfg, device)
        print(f"  val_loss: {val_metrics['loss']:.4f} | "
              f"AUPRC donor: {val_metrics['auprc_donor']:.4f} | "
              f"AUPRC acc: {val_metrics['auprc_acceptor']:.4f} | "
              f"AUPRC mean: {val_metrics['auprc_mean']:.4f}")

        wandb.log({
            "val/loss_total": val_metrics["loss"],
            "val/auprc_donor": val_metrics["auprc_donor"],
            "val/auprc_acceptor": val_metrics["auprc_acceptor"],
            "val/auprc_mean": val_metrics["auprc_mean"],
            **{f"val/{k}": v for k, v in val_metrics.items()
               if k.startswith("topk_")},
            "epoch": epoch,
        }, step=global_step)

        # Checkpointing
        ckpt_state = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "best_auprc": best_auprc,
            "patience_counter": patience_counter,
            "config": cfg,
        }

        # Save last checkpoint
        torch.save(ckpt_state, ckpt_dir / "last.pt")

        # Save best checkpoint
        if val_metrics["auprc_mean"] > best_auprc:
            best_auprc = val_metrics["auprc_mean"]
            patience_counter = 0
            ckpt_state["best_auprc"] = best_auprc
            torch.save(ckpt_state, ckpt_dir / "best.pt")
            print(f"  New best AUPRC: {best_auprc:.4f} — saved best.pt")
        else:
            patience_counter += 1
            print(f"  No improvement ({patience_counter}/{cfg['early_stopping_patience']})")

        # Early stopping
        if patience_counter >= cfg["early_stopping_patience"]:
            print(f"Early stopping at epoch {epoch} (patience exhausted)")
            break

    wandb.finish()
    print(f"Training complete. Best validation AUPRC: {best_auprc:.4f}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SpliceMamba")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    args = parser.parse_args()
    train(CONFIG, resume_path=args.resume)
