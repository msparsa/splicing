"""
SpliceMamba v5 model architecture.

Same encoder as v4 (DeepConvStem + SinusoidalPosEnc + BiMamba2). The
sliding-window self-attention block is replaced by a Top-N cross-attention
block driven by the coarse head's predictions:

  * Coarse head emits (B, L, 3) softmax over {neither, acceptor, donor}.
  * Top-N donor and Top-N acceptor positions are selected per sample inside
    the labeled region [5000:10000], with optional Gumbel noise during
    training (Gumbel-Top-K, Kool et al. 2019).
  * Each selected center is expanded to a +/-vicinity_radius window (default
    +/-100bp), and the union of these positions (deduped) forms the Q stream.
  * Cross-attention: Q = vicinity embeddings, KV = the full encoder output
    (frozen across cross-attn layers).
  * Refined Q values are scattered back to their original positions; positions
    not selected pass through from the encoder output.

Forward signature matches v4: (coarse_logits, refined_logits, encoder_output).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from flash_attn import flash_attn_func

from model import (
    DeepConvStem,
    SinusoidalPositionalEncoding,
    BiMambaEncoder,
    ClassificationHead,
    DropPath,
)


# ---------------------------------------------------------------------------
# Gumbel-Top-K vicinity selection
# ---------------------------------------------------------------------------

def select_vicinities_gumbel(
    coarse_logits: torch.Tensor,
    top_n: int,
    vicinity_radius: int,
    label_start: int,
    label_end: int,
    seq_len: int,
    gumbel_tau: float,
    training: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select Top-N donor and Top-N acceptor vicinities per sample.

    Parameters
    ----------
    coarse_logits   : (B, L, 3)  logits from the coarse head
    top_n           : number of donor and acceptor centers per sample
    vicinity_radius : half-width of the vicinity window around each center
    label_start/end : range [label_start, label_end) is the only region
                      where centers may be selected
    seq_len         : full sequence length L (sentinel index for pad slots)
    gumbel_tau      : additive Gumbel-noise scale during training (0 = none)
    training        : whether the model is in training mode

    Returns
    -------
    vicinity_idx : (B, Q_max) long. Q_max = 2 * top_n * (2*r+1).
                   Real positions are in [0, L-1]; pad / duplicate slots
                   carry the sentinel value `seq_len` (= L), which points
                   at the L-row in the (B, L+1, D) scratch buffer used by
                   the caller.
    q_pad_mask   : (B, Q_max) bool. True at unique real positions, False at
                   pad / duplicate slots.
    """
    B, L, _ = coarse_logits.shape
    device = coarse_logits.device

    # Use log-softmax instead of softmax+log for numerical stability.
    log_probs = F.log_softmax(coarse_logits.float(), dim=-1)
    log_p_acc = log_probs[..., 1]   # (B, L)
    log_p_don = log_probs[..., 2]   # (B, L)

    # Restrict candidate centers to the labeled region.
    region_mask = torch.zeros(L, dtype=torch.bool, device=device)
    region_mask[label_start:label_end] = True
    log_p_acc = log_p_acc.masked_fill(~region_mask, float("-inf"))
    log_p_don = log_p_don.masked_fill(~region_mask, float("-inf"))

    # Gumbel-Top-K (training only).
    if training and gumbel_tau > 0.0:
        # Standard Gumbel(0, 1): g = -log(-log(U)).
        u_acc = torch.rand_like(log_p_acc).clamp_(min=1e-12)
        u_don = torch.rand_like(log_p_don).clamp_(min=1e-12)
        g_acc = -torch.log(-torch.log(u_acc))
        g_don = -torch.log(-torch.log(u_don))
        scores_acc = log_p_acc + gumbel_tau * g_acc
        scores_don = log_p_don + gumbel_tau * g_don
    else:
        scores_acc = log_p_acc
        scores_don = log_p_don

    # Top-N centers per type per sample.
    don_idx = scores_don.topk(top_n, dim=-1).indices   # (B, N)
    acc_idx = scores_acc.topk(top_n, dim=-1).indices   # (B, N)
    centers = torch.cat([don_idx, acc_idx], dim=-1)    # (B, 2N)

    # Expand each center to +/- vicinity_radius.
    offsets = torch.arange(
        -vicinity_radius, vicinity_radius + 1, device=device
    )                                                  # (2r+1,)
    vicinity = centers.unsqueeze(-1) + offsets         # (B, 2N, 2r+1)
    vicinity = vicinity.clamp_(0, L - 1).reshape(B, -1)  # (B, Q_max)

    # Per-sample dedup: sort, mark first occurrences, replace duplicates with
    # the sentinel `L` (out-of-range; gather/scatter caller uses L+1 buffer).
    sorted_idx, _ = vicinity.sort(dim=-1)
    is_first = torch.ones_like(sorted_idx, dtype=torch.bool)
    is_first[:, 1:] = sorted_idx[:, 1:] != sorted_idx[:, :-1]
    vicinity_idx = torch.where(
        is_first, sorted_idx, torch.full_like(sorted_idx, seq_len)
    )

    return vicinity_idx, is_first


# ---------------------------------------------------------------------------
# Selection diagnostics (W&B logging)
# ---------------------------------------------------------------------------

@torch.no_grad()
def selection_diagnostics(
    coarse_logits: torch.Tensor,
    vicinity_idx: torch.Tensor,
    q_pad_mask: torch.Tensor,
    y: torch.Tensor,
    label_start: int,
    label_end: int,
    seq_len: int,
) -> dict:
    """Compute Gumbel-Top-K selection-quality metrics.

    Parameters
    ----------
    coarse_logits : (B, L, 3) coarse-head logits used for selection
    vicinity_idx  : (B, Q_max) long, sentinel = `seq_len` at pad / dup slots
    q_pad_mask    : (B, Q_max) bool, True at unique real positions
    y             : (B, label_end - label_start) ground-truth labels in
                    {0=neither, 1=acceptor, 2=donor}

    Returns
    -------
    dict of scalar floats keyed under "selection/...":
      recall_donor / recall_acceptor — fraction of true donors / acceptors
        whose position is contained in the dedup'd vicinity set
      coverage_label_region — fraction of [label_start, label_end) positions
        included in the vicinity set, averaged over the batch
      dedup_frac — real_q / Q_max, averaged over the batch (1.0 = no overlap;
        lower = more vicinity overlap was deduped away)
      mean_p_donor_at_selected / mean_p_acceptor_at_selected — average
        coarse-head donor / acceptor softmax probability at selected
        (vicinity) positions
    """
    B, L, _ = coarse_logits.shape
    device = coarse_logits.device

    # Build per-sample membership mask: True at every selected position.
    # Use an (L+1) buffer to absorb sentinel writes; slice off the last col.
    membership = torch.zeros(B, L + 1, dtype=torch.bool, device=device)
    membership.scatter_(1, vicinity_idx, q_pad_mask.to(torch.bool))
    membership = membership[:, :L]

    in_label = membership[:, label_start:label_end]      # (B, label_len)
    is_donor = (y == 2)
    is_acc = (y == 1)
    n_donor = is_donor.sum().clamp(min=1).float()
    n_acc = is_acc.sum().clamp(min=1).float()
    recall_donor = (in_label & is_donor).sum().float() / n_donor
    recall_acc = (in_label & is_acc).sum().float() / n_acc

    coverage = in_label.float().mean()
    dedup_frac = q_pad_mask.float().mean()

    # Mean coarse-head confidence at unique selected positions.
    probs = coarse_logits.softmax(-1)
    real_idx = vicinity_idx.clamp_max(L - 1)             # sentinels masked below
    p_don_sel = probs[..., 2].gather(1, real_idx)        # (B, Q_max)
    p_acc_sel = probs[..., 1].gather(1, real_idx)
    real_count = q_pad_mask.float().sum().clamp(min=1)
    mean_p_donor = (p_don_sel * q_pad_mask.float()).sum() / real_count
    mean_p_acc = (p_acc_sel * q_pad_mask.float()).sum() / real_count

    return {
        "selection/recall_donor": recall_donor.item(),
        "selection/recall_acceptor": recall_acc.item(),
        "selection/coverage_label_region": coverage.item(),
        "selection/dedup_frac": dedup_frac.item(),
        "selection/mean_p_donor_at_selected": mean_p_donor.item(),
        "selection/mean_p_acceptor_at_selected": mean_p_acc.item(),
    }


# ---------------------------------------------------------------------------
# Top-N Cross-Attention Layer
# ---------------------------------------------------------------------------

class TopNCrossAttention(nn.Module):
    """Pre-norm cross-attention + FFN.

    Q stream attends to a fixed (frozen-across-layers) KV memory derived from
    the encoder output. Pad rows in the Q stream compute attention but their
    contributions are masked to zero after each sublayer so they don't affect
    the eventual scatter-back.
    """

    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 8,
        dropout: float = 0.1,
        ffn_expand: int = 4,
        drop_path: float = 0.0,
    ):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        # Cross-attention sublayer
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.drop_path_attn = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        # FFN sublayer
        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * ffn_expand),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * ffn_expand, d_model),
            nn.Dropout(dropout),
        )
        self.drop_path_ffn = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(
        self,
        kv: torch.Tensor,
        q_stream: torch.Tensor,
        q_pad_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        kv         : (B, L, D)  encoder output (KV memory)
        q_stream   : (B, Q, D)  current Q-stream embeddings (pad rows = 0)
        q_pad_mask : (B, Q) bool — True at real positions, False at pad slots

        Returns
        -------
        (B, Q, D) refined Q-stream with pad rows zeroed.
        """
        B, Q, D = q_stream.shape
        L = kv.size(1)
        H, Hd = self.n_heads, self.head_dim

        # ---- Cross-attention sublayer ----
        q_in = self.norm_q(q_stream)
        kv_in = self.norm_kv(kv)

        q = self.q_proj(q_in).view(B, Q, H, Hd)
        k = self.k_proj(kv_in).view(B, L, H, Hd)
        v = self.v_proj(kv_in).view(B, L, H, Hd)

        # FlashAttention-2 supports cross-attention with different Q and K
        # seqlens out of the box. Pad rows attend over the full KV; their
        # outputs are zeroed below before they re-enter any computation.
        attn_out = flash_attn_func(q, k, v, causal=False)   # (B, Q, H, Hd)
        attn_out = attn_out.reshape(B, Q, D)

        q_stream = q_stream + self.drop_path_attn(
            self.dropout(self.out_proj(attn_out))
        )
        q_stream = q_stream.masked_fill(~q_pad_mask.unsqueeze(-1), 0.0)

        # ---- FFN sublayer ----
        q_stream = q_stream + self.drop_path_ffn(self.ffn(self.ffn_norm(q_stream)))
        q_stream = q_stream.masked_fill(~q_pad_mask.unsqueeze(-1), 0.0)

        return q_stream


# ---------------------------------------------------------------------------
# Top-level SpliceMambaV5
# ---------------------------------------------------------------------------

class SpliceMambaV5(nn.Module):
    """
    SpliceMamba v5 — Top-N cross-attention variant.

    Forward returns (coarse_logits, refined_logits, encoder_output) with the
    same shapes as v4, so train.py / evaluate.py pipelines downstream do not
    need changes beyond model instantiation.
    """

    def __init__(
        self,
        d_model: int = 256,
        n_mamba_layers: int = 8,
        d_state: int = 64,
        expand: int = 2,
        d_conv: int = 4,
        headdim: int = 32,
        n_cross_attn_layers: int = 2,
        n_heads: int = 8,
        top_n: int = 20,
        vicinity_radius: int = 100,
        gumbel_tau: float = 1.0,
        coarse_select_in_label_only: bool = True,
        label_start: int = 5000,
        label_end: int = 10000,
        dropout: float = 0.15,
        drop_path_rate: float = 0.0,
        n_classes: int = 3,
        max_len: int = 15000,
    ):
        super().__init__()
        self.d_model = d_model
        self.top_n = top_n
        self.vicinity_radius = vicinity_radius
        self.gumbel_tau = gumbel_tau
        self.coarse_select_in_label_only = coarse_select_in_label_only
        self.label_start = label_start
        self.label_end = label_end
        self.max_len = max_len
        self.q_max = 2 * top_n * (2 * vicinity_radius + 1)

        self.stem = DeepConvStem(d_model)
        self.pos_enc = SinusoidalPositionalEncoding(d_model, max_len)
        self.encoder = BiMambaEncoder(
            n_layers=n_mamba_layers,
            d_model=d_model,
            d_state=d_state,
            expand=expand,
            d_conv=d_conv,
            headdim=headdim,
            dropout=dropout,
            drop_path_rate=drop_path_rate,
        )
        self.coarse_head = ClassificationHead(d_model, n_classes, dropout)

        attn_dp_rates = [
            drop_path_rate * i / max(n_cross_attn_layers - 1, 1)
            for i in range(n_cross_attn_layers)
        ]
        self.cross_attn_layers = nn.ModuleList([
            TopNCrossAttention(
                d_model=d_model,
                n_heads=n_heads,
                dropout=dropout,
                drop_path=attn_dp_rates[i],
            )
            for i in range(n_cross_attn_layers)
        ])
        self.refined_head = ClassificationHead(d_model, n_classes, dropout)

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x : (B, 4, L) one-hot DNA

        Returns
        -------
        coarse_logits  : (B, L, 3)
        refined_logits : (B, L, 3)
        encoder_output : (B, L, D)
        """
        B = x.size(0)
        D = self.d_model
        L = self.max_len

        # Stem + positional encoding + bidirectional Mamba2 encoder
        h = self.stem(x)
        h = self.pos_enc(h)
        h = self.encoder(h)
        encoder_output = h                                  # (B, L, D)

        # Coarse head (auxiliary loss + selector for cross-attention)
        coarse_logits = self.coarse_head(h)                 # (B, L, 3)

        if self.coarse_select_in_label_only:
            label_start, label_end = self.label_start, self.label_end
        else:
            label_start, label_end = 0, L

        # Selection is non-differentiable: compute under no_grad. Gradients
        # to the coarse head come from the coarse CE loss directly.
        with torch.no_grad():
            vicinity_idx, q_pad_mask = select_vicinities_gumbel(
                coarse_logits,
                top_n=self.top_n,
                vicinity_radius=self.vicinity_radius,
                label_start=label_start,
                label_end=label_end,
                seq_len=L,
                gumbel_tau=self.gumbel_tau,
                training=self.training,
            )                                               # (B, Q), (B, Q)
            # Stash for diagnostics (consumed by train.py / validate()).
            self._last_selection = (vicinity_idx, q_pad_mask)

        # Build a (B, L+1, D) scratch buffer with the L-row zeroed; the pad
        # sentinel `L` then points at this zero row.
        scratch = F.pad(encoder_output, (0, 0, 0, 1))       # (B, L+1, D)

        # Initial Q stream: gather encoder embeddings at vicinity positions.
        q_stream = scratch.gather(
            1, vicinity_idx.unsqueeze(-1).expand(-1, -1, D)
        )                                                   # (B, Q, D)

        # Stack of cross-attention layers (KV memory is fixed = encoder_output).
        for layer in self.cross_attn_layers:
            q_stream = layer(encoder_output, q_stream, q_pad_mask)

        # Scatter refined Q values back into the scratch buffer. Pad rows
        # write to row L which we slice off before the refined head.
        out_scratch = scratch.clone()
        out_scratch.scatter_(
            1,
            vicinity_idx.unsqueeze(-1).expand(-1, -1, D),
            q_stream,
        )
        refined = out_scratch[:, :L]                        # (B, L, D)

        refined_logits = self.refined_head(refined)         # (B, L, 3)
        return coarse_logits, refined_logits, encoder_output

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq-len", type=int, default=15000)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--vicinity-radius", type=int, default=100)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SpliceMambaV5(
        d_model=256,
        n_heads=8,
        top_n=args.top_n,
        vicinity_radius=args.vicinity_radius,
        n_cross_attn_layers=2,
        max_len=args.seq_len,
    ).to(device)
    print(f"SpliceMambaV5 parameters: {model.count_parameters():,}")
    print(f"Q_max = {model.q_max}")

    x = torch.randn(args.batch, 4, args.seq_len, device=device)

    # ---- forward shapes under bf16 autocast ----
    model.train()
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        coarse, refined, enc = model(x)
    assert coarse.shape == (args.batch, args.seq_len, 3), coarse.shape
    assert refined.shape == (args.batch, args.seq_len, 3), refined.shape
    assert enc.shape == (args.batch, args.seq_len, 256), enc.shape
    print("forward shapes OK")

    # ---- training-mode stochasticity (Gumbel noise active) ----
    with torch.no_grad():
        v1, _ = select_vicinities_gumbel(
            coarse, model.top_n, model.vicinity_radius,
            model.label_start, model.label_end, model.max_len,
            gumbel_tau=model.gumbel_tau, training=True,
        )
        v2, _ = select_vicinities_gumbel(
            coarse, model.top_n, model.vicinity_radius,
            model.label_start, model.label_end, model.max_len,
            gumbel_tau=model.gumbel_tau, training=True,
        )
    assert not torch.equal(v1, v2), "training selection should be stochastic"
    print("training stochasticity OK")

    # ---- eval-mode determinism ----
    model.eval()
    with torch.no_grad():
        v1, _ = select_vicinities_gumbel(
            coarse, model.top_n, model.vicinity_radius,
            model.label_start, model.label_end, model.max_len,
            gumbel_tau=model.gumbel_tau, training=False,
        )
        v2, _ = select_vicinities_gumbel(
            coarse, model.top_n, model.vicinity_radius,
            model.label_start, model.label_end, model.max_len,
            gumbel_tau=model.gumbel_tau, training=False,
        )
    assert torch.equal(v1, v2), "eval selection must be deterministic"
    print("eval determinism OK")

    # ---- gradient flow ----
    model.train()
    x = torch.randn(args.batch, 4, args.seq_len, device=device, requires_grad=False)
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        coarse, refined, _ = model(x)
    loss = coarse.float().sum() + refined.float().sum()
    loss.backward()
    assert model.coarse_head.net[0].weight.grad is not None, "coarse head no grad"
    assert model.cross_attn_layers[0].q_proj.weight.grad is not None, "q_proj no grad"
    assert model.encoder.fwd_layers[-1].mamba.in_proj.weight.grad is not None, "encoder no grad"
    print("gradient flow OK")
    print("all smoke tests passed")
