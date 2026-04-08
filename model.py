"""
SpliceMamba v3 model architecture.

Deep Dilated Conv Stem → Sinusoidal Positional Encoding → BiMamba2 Encoder
→ Coarse Head → Sliding-Window Attention (with FFN) → Refined Head

Key changes from v2:
- Deep dilated conv stem (4 residual blocks, dilations 1/4/10/25) replaces
  shallow 2-branch stem for better local motif capture (~641bp receptive field)
- Sliding-window attention now includes standard transformer FFN sublayers
- Coarse loss weight reduced (0.3 → 0.1) to free encoder representations
"""

from __future__ import annotations


import math
import torch
import torch.nn as nn
from mamba_ssm import Mamba2
from flash_attn import flash_attn_func


# ---------------------------------------------------------------------------
# Deep Dilated Conv Stem
# ---------------------------------------------------------------------------

class DilatedResBlock(nn.Module):
    """Dilated residual conv block: Conv→BN→GELU→Conv→BN + residual → GELU.

    Two convolutions with the same dilation rate, wrapped in a residual
    connection.  Captures patterns at a specific spatial scale.
    """

    def __init__(self, channels: int, kernel_size: int = 5, dilation: int = 1):
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.conv1 = nn.Conv1d(channels, channels, kernel_size,
                               dilation=dilation, padding=padding)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size,
                               dilation=dilation, padding=padding)
        self.bn2 = nn.BatchNorm1d(channels)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.act(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return self.act(x + residual)


class DeepConvStem(nn.Module):
    """Stacked dilated residual blocks for multi-scale local motif capture.

    Dilations [1, 4, 10, 25] give an effective receptive field of ~641bp,
    spanning local splice motifs (GT/AG), branch points (~20-40bp),
    polypyrimidine tracts (~30bp), and ESE/ISS elements (~100bp).
    """

    def __init__(self, d_model: int = 256):
        super().__init__()
        self.input_conv = nn.Sequential(
            nn.Conv1d(4, d_model, kernel_size=11, padding=5),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(
            DilatedResBlock(d_model, kernel_size=5, dilation=1),
            DilatedResBlock(d_model, kernel_size=5, dilation=4),
            DilatedResBlock(d_model, kernel_size=5, dilation=10),
            DilatedResBlock(d_model, kernel_size=5, dilation=25),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="linear")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : (B, 4, L) → (B, L, D)"""
        h = self.input_conv(x)  # (B, D, L)
        h = self.blocks(h)      # (B, D, L)
        return h.transpose(1, 2)  # (B, L, D)


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
# Mamba2 Layer with pre-norm and residual
# ---------------------------------------------------------------------------

class Mamba2Layer(nn.Module):
    """LayerNorm → Mamba2 → Dropout → residual."""

    def __init__(self, d_model: int, d_state: int, expand: int, d_conv: int,
                 headdim: int = 32, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.mamba = Mamba2(
            d_model=d_model,
            d_state=d_state,
            expand=expand,
            d_conv=d_conv,
            headdim=headdim,
            bias=False,
            conv_bias=True,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.dropout(self.mamba(self.norm(x)))


# ---------------------------------------------------------------------------
# Bidirectional Mamba2 Encoder
# ---------------------------------------------------------------------------

class BiMambaEncoder(nn.Module):
    """Two independent Mamba2 stacks (forward + backward), fused by projection."""

    def __init__(
        self,
        n_layers: int = 8,
        d_model: int = 256,
        d_state: int = 64,
        expand: int = 2,
        d_conv: int = 4,
        headdim: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.fwd_layers = nn.ModuleList([
            Mamba2Layer(d_model, d_state, expand, d_conv, headdim, dropout)
            for _ in range(n_layers)
        ])
        self.bwd_layers = nn.ModuleList([
            Mamba2Layer(d_model, d_state, expand, d_conv, headdim, dropout)
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
# Sliding-Window Self-Attention Layer (standard pre-norm residual)
# ---------------------------------------------------------------------------

class SlidingWindowAttention(nn.Module):
    """Pre-norm sliding-window self-attention + FFN, standard transformer block."""

    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 8,
        window_radius: int = 400,
        dropout: float = 0.1,
        ffn_expand: int = 4,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.window_radius = window_radius

        # Attention sublayer
        self.norm = nn.LayerNorm(d_model)
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

        # FFN sublayer
        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * ffn_expand),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * ffn_expand, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : (B, L, D) → (B, L, D)"""
        B, L, D = x.shape

        # Attention sublayer
        h = self.norm(x)
        q = self.q_proj(h).view(B, L, self.n_heads, self.head_dim)
        k = self.k_proj(h).view(B, L, self.n_heads, self.head_dim)
        v = self.v_proj(h).view(B, L, self.n_heads, self.head_dim)

        # FlashAttention-2 expects (B, L, H, d) layout
        attn_out = flash_attn_func(
            q, k, v,
            window_size=(self.window_radius, self.window_radius),
            causal=False,
        )  # (B, L, H, d)

        attn_out = attn_out.reshape(B, L, D)
        x = x + self.dropout(self.out_proj(attn_out))

        # FFN sublayer
        x = x + self.ffn(self.ffn_norm(x))

        return x


# ---------------------------------------------------------------------------
# Top-level SpliceMamba
# ---------------------------------------------------------------------------

class SpliceMamba(nn.Module):
    """
    Full SpliceMamba v3 model.

    Forward returns (coarse_logits, refined_logits, encoder_output).
    """

    def __init__(
        self,
        d_model: int = 256,
        n_mamba_layers: int = 8,
        d_state: int = 64,
        expand: int = 2,
        d_conv: int = 4,
        headdim: int = 32,
        n_attn_layers: int = 4,
        n_heads: int = 8,
        window_radius: int = 400,
        dropout: float = 0.1,
        n_classes: int = 3,
        max_len: int = 15000,
    ):
        super().__init__()
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
        )
        self.coarse_head = ClassificationHead(d_model, n_classes, dropout)
        self.attn_layers = nn.ModuleList([
            SlidingWindowAttention(d_model, n_heads, window_radius, dropout)
            for _ in range(n_attn_layers)
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
        encoder_output : (B, L, D)  — for Phase 2 heads
        """
        # Stem + positional encoding
        h = self.stem(x)          # (B, L, D)
        h = self.pos_enc(h)       # (B, L, D)

        # Bidirectional Mamba2 encoder
        h = self.encoder(h)       # (B, L, D)
        encoder_output = h

        # Coarse head (auxiliary loss, decoupled from attention)
        coarse_logits = self.coarse_head(h)  # (B, L, 3)

        # Sliding-window attention refinement
        attn_input = h
        for attn_layer in self.attn_layers:
            attn_input = attn_layer(attn_input)

        # Refined head
        refined_logits = self.refined_head(attn_input)  # (B, L, 3)

        return coarse_logits, refined_logits, encoder_output

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
