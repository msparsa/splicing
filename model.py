"""
SpliceMamba model architecture.

Conv Stem → Sinusoidal Positional Encoding → BiMamba Encoder
→ Coarse Head → Gated Sliding-Window Attention → Refined Head

~8M parameters. Operates at nucleotide resolution (no downsampling).
"""

from __future__ import annotations


import math
import torch
import torch.nn as nn
from mamba_ssm import Mamba
from flash_attn import flash_attn_func


# ---------------------------------------------------------------------------
# Conv Stem
# ---------------------------------------------------------------------------

class ConvStem(nn.Module):
    """Project 4-channel one-hot DNA to D=256, capturing local motifs."""

    def __init__(self, d_model: int = 256):
        super().__init__()
        self.conv1 = nn.Conv1d(4, 128, kernel_size=11, stride=1, padding=5)
        self.bn1 = nn.BatchNorm1d(128)
        self.conv2 = nn.Conv1d(128, d_model, kernel_size=1, stride=1)
        self.bn2 = nn.BatchNorm1d(d_model)
        self.act = nn.GELU()
        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_normal_(self.conv1.weight, nonlinearity="linear")
        nn.init.zeros_(self.conv1.bias)
        nn.init.kaiming_normal_(self.conv2.weight, nonlinearity="linear")
        nn.init.zeros_(self.conv2.bias)
        nn.init.ones_(self.bn1.weight)
        nn.init.zeros_(self.bn1.bias)
        nn.init.ones_(self.bn2.weight)
        nn.init.zeros_(self.bn2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, 4, L)  → output : (B, L, D)
        """
        x = self.act(self.bn1(self.conv1(x)))
        x = self.act(self.bn2(self.conv2(x)))
        return x.transpose(1, 2)  # (B, L, D)


# ---------------------------------------------------------------------------
# Sinusoidal Positional Encoding
# ---------------------------------------------------------------------------

class SinusoidalPositionalEncoding(nn.Module):
    """Vaswani et al. 2017 sinusoidal positional encoding, registered as buffer."""

    def __init__(self, d_model: int = 256, max_len: int = 15000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, D)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : (B, L, D) → (B, L, D)"""
        return x + self.pe[:, : x.size(1)]


# ---------------------------------------------------------------------------
# Mamba Layer with pre-norm and residual
# ---------------------------------------------------------------------------

class MambaLayer(nn.Module):
    """LayerNorm → Mamba → Dropout → residual."""

    def __init__(self, d_model: int, d_state: int, expand: int, d_conv: int,
                 dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            expand=expand,
            d_conv=d_conv,
            bias=False,
            conv_bias=True,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.dropout(self.mamba(self.norm(x)))


# ---------------------------------------------------------------------------
# Bidirectional Mamba Encoder
# ---------------------------------------------------------------------------

class BiMambaEncoder(nn.Module):
    """Two independent Mamba stacks (forward + backward), fused by projection."""

    def __init__(
        self,
        n_layers: int = 6,
        d_model: int = 256,
        d_state: int = 16,
        expand: int = 1,
        d_conv: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.fwd_layers = nn.ModuleList([
            MambaLayer(d_model, d_state, expand, d_conv, dropout)
            for _ in range(n_layers)
        ])
        self.bwd_layers = nn.ModuleList([
            MambaLayer(d_model, d_state, expand, d_conv, dropout)
            for _ in range(n_layers)
        ])
        self.fusion = nn.Linear(2 * d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : (B, L, D) → (B, L, D)"""
        # Forward pass (left-to-right)
        h_fwd = x
        for layer in self.fwd_layers:
            h_fwd = layer(h_fwd)

        # Backward pass (right-to-left via reversal)
        h_bwd = x.flip(1)
        for layer in self.bwd_layers:
            h_bwd = layer(h_bwd)
        h_bwd = h_bwd.flip(1)  # reverse back

        # Fuse
        h = self.fusion(torch.cat([h_fwd, h_bwd], dim=-1))
        return h


# ---------------------------------------------------------------------------
# Classification Head
# ---------------------------------------------------------------------------

class ClassificationHead(nn.Module):
    """Two-layer MLP: D → D → 3 classes."""

    def __init__(self, d_model: int = 256, n_classes: int = 3, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Gated Sliding-Window Self-Attention Layer
# ---------------------------------------------------------------------------

class GatedSlidingWindowAttention(nn.Module):
    """One layer of gated local self-attention using FlashAttention-2."""

    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 8,
        window_radius: int = 200,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.window_radius = window_radius

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        gate: torch.Tensor,
    ) -> torch.Tensor:
        """
        x    : (B, L, D) — input features
        gate : (B, L, 1) — per-position gate values in [0, 1]
        """
        B, L, D = x.shape

        q = self.q_proj(x).view(B, L, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(B, L, self.n_heads, self.head_dim)
        v = self.v_proj(x).view(B, L, self.n_heads, self.head_dim)

        # FlashAttention-2 expects (B, L, H, d) layout
        attn_out = flash_attn_func(
            q, k, v,
            window_size=(self.window_radius, self.window_radius),
            causal=False,
        )  # (B, L, H, d)

        attn_out = attn_out.reshape(B, L, D)
        attn_out = self.dropout(self.out_proj(attn_out))

        # Gated residual: input × (1 - gate) + attn_out × gate
        output = self.norm(x * (1 - gate) + attn_out * gate)
        return output


# ---------------------------------------------------------------------------
# Top-level SpliceMamba
# ---------------------------------------------------------------------------

class SpliceMamba(nn.Module):
    """
    Full SpliceMamba model.

    Forward returns (coarse_logits, refined_logits, encoder_output).
    """

    def __init__(
        self,
        d_model: int = 256,
        n_mamba_layers: int = 6,
        d_state: int = 16,
        expand: int = 1,
        d_conv: int = 4,
        n_attn_layers: int = 2,
        n_heads: int = 8,
        window_radius: int = 200,
        dropout: float = 0.1,
        n_classes: int = 3,
        max_len: int = 15000,
    ):
        super().__init__()
        self.stem = ConvStem(d_model)
        self.pos_enc = SinusoidalPositionalEncoding(d_model, max_len)
        self.encoder = BiMambaEncoder(
            n_layers=n_mamba_layers,
            d_model=d_model,
            d_state=d_state,
            expand=expand,
            d_conv=d_conv,
            dropout=dropout,
        )
        self.coarse_head = ClassificationHead(d_model, n_classes, dropout)
        self.attn_layers = nn.ModuleList([
            GatedSlidingWindowAttention(d_model, n_heads, window_radius, dropout)
            for _ in range(n_attn_layers)
        ])
        self.refined_head = ClassificationHead(d_model, n_classes, dropout)

    def forward(
        self,
        x: torch.Tensor,
        tau: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x   : (B, 4, L) one-hot DNA
        tau : gate temperature (annealed during training)

        Returns
        -------
        coarse_logits  : (B, L, 3)
        refined_logits : (B, L, 3)
        encoder_output : (B, L, D)  — for Phase 2 heads
        """
        # Stem + positional encoding
        h = self.stem(x)          # (B, L, D)
        h = self.pos_enc(h)       # (B, L, D)

        # Bidirectional Mamba encoder
        h = self.encoder(h)       # (B, L, D)
        encoder_output = h

        # Coarse head
        coarse_logits = self.coarse_head(h)  # (B, L, 3)

        # Gate from coarse logits
        splice_logits = torch.max(
            coarse_logits[:, :, 1], coarse_logits[:, :, 2]
        )  # (B, L)
        gate = torch.sigmoid(splice_logits / tau).unsqueeze(-1)  # (B, L, 1)

        # Gated sliding-window attention (2 layers)
        attn_input = h
        for attn_layer in self.attn_layers:
            attn_input = attn_layer(attn_input, gate)

        # Refined head
        refined_logits = self.refined_head(attn_input)  # (B, L, 3)

        return coarse_logits, refined_logits, encoder_output

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
