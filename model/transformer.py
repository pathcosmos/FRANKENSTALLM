"""
Full transformer: TransformerBlock and top-level LLM model.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import LMConfig
from .layers import RMSNorm, RotaryEmbedding, SwiGLU
from .attention import MultiHeadAttention

# ---------------------------------------------------------------------------
# Optional TransformerEngine import (FP8 support)
# ---------------------------------------------------------------------------
try:
    import transformer_engine.pytorch as te  # type: ignore[import]
    HAS_TE = True
except ImportError:
    te = None  # type: ignore[assignment]
    HAS_TE = False


# ---------------------------------------------------------------------------
# Transformer Block
# ---------------------------------------------------------------------------

class TransformerBlock(nn.Module):
    """Single pre-norm transformer decoder block.

    Layout:
        x = x + Attention( RMSNorm(x) )
        x = x + FFN( RMSNorm(x) )
    """

    def __init__(self, config: LMConfig) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(config.d_model)
        self.attn      = MultiHeadAttention(config)
        self._use_fp8  = config.use_fp8 and HAS_TE

        if self._use_fp8:
            # te.LayerNormMLP fuses RMSNorm + gate/up/down projections into one kernel.
            # It applies normalisation internally, so ffn_norm is not needed.
            self.ffn_norm = None
            self.ffn = te.LayerNormMLP(
                hidden_size=config.d_model,
                ffn_hidden_size=config.d_ffn,
                bias=config.bias,
                activation="swiglu",
                normalization="RMSNorm",
            )
        else:
            self.ffn_norm = RMSNorm(config.d_model)
            self.ffn      = SwiGLU(config.d_model, config.d_ffn, bias=config.bias)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x:   (B, T, C)
            cos: (T, head_dim // 2)
            sin: (T, head_dim // 2)

        Returns:
            (B, T, C)
        """
        # Pre-norm attention with residual
        x = x + self.attn(self.attn_norm(x), cos, sin)
        # FFN with residual — te.LayerNormMLP applies norm internally
        if self._use_fp8:
            x = x + self.ffn(x)
        else:
            x = x + self.ffn(self.ffn_norm(x))
        return x


# ---------------------------------------------------------------------------
# Full Language Model
# ---------------------------------------------------------------------------

class LLM(nn.Module):
    """Decoder-only transformer language model.

    Features:
    - Learned token embeddings with weight tying to the LM head
    - Rotary positional embeddings (no learned position embeddings)
    - Stack of pre-norm TransformerBlocks
    - Final RMSNorm before the LM head
    - Optional cross-entropy loss computation (for training)
    """

    def __init__(self, config: LMConfig) -> None:
        super().__init__()
        self.config = config

        # --- Embedding -------------------------------------------------------
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)

        # --- Transformer layers ----------------------------------------------
        self.layers = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.n_layers)]
        )

        # --- Final normalisation and LM head ---------------------------------
        self.norm    = RMSNorm(config.d_model)
        # NOTE: lm_head는 nn.Linear 유지 — embedding weight tying + TE FP8 호환성
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Weight tying: share embedding and LM-head weight matrices
        self.lm_head.weight = self.embedding.weight

        # --- Rotary embeddings -----------------------------------------------
        self.rope = RotaryEmbedding(
            dim=config.head_dim,
            max_seq_len=config.max_seq_len,
            theta=config.rope_theta,
        )

        # --- Initialise weights ----------------------------------------------
        self.apply(self._init_weights)

    # ------------------------------------------------------------------
    # Weight initialisation
    # ------------------------------------------------------------------

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        """Apply standard initialisation:
        - Linear / Embedding weights: N(0, 0.02)
        - Bias parameters: zeros
        - te.Linear / te.LayerNormMLP: skipped (TE manages its own init)
        """
        # TE modules handle their own weight initialisation.
        if HAS_TE and isinstance(module, (te.Linear, te.LayerNormMLP)):
            return
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            input_ids: (B, T) long tensor of token indices
            targets:   (B, T) long tensor of target token indices, or None.
                       Use -1 (ignore_index) to mask positions.

        Returns:
            logits: (B, T, vocab_size)
            loss:   scalar cross-entropy loss, or None if targets is None
        """
        B, T = input_ids.shape
        device = input_ids.device

        # Token embeddings: (B, T, C)
        x = self.embedding(input_ids)

        # Rotary cos/sin for this sequence length: (T, head_dim // 2)
        cos, sin = self.rope(T, device)

        # Run through transformer blocks
        for layer in self.layers:
            x = layer(x, cos, sin)

        # Final normalisation
        x = self.norm(x)

        # LM head: (B, T, vocab_size)
        logits = self.lm_head(x)

        # Compute loss if targets are provided
        loss: Optional[torch.Tensor] = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
            )

        return logits, loss

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def num_params(self) -> int:
        """Number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_input_embeddings(self) -> nn.Embedding:
        """HuggingFace-compatible accessor for the token embedding layer."""
        return self.embedding

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, config: LMConfig) -> "LLM":
        """Construct an LLM from an LMConfig instance."""
        return cls(config)

    @classmethod
    def from_pretrained(cls, path: str | Path) -> "LLM":
        """Load model from a checkpoint directory.

        Expects:
            <path>/config.yaml  — serialised LMConfig
            <path>/model.pt     — state dict produced by save_pretrained
        """
        path = Path(path)
        config = LMConfig.from_yaml(path / "config.yaml")
        model = cls(config)
        state_dict = torch.load(
            path / "model.pt",
            map_location="cpu",
            weights_only=True,
        )
        model.load_state_dict(state_dict)
        return model

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_pretrained(self, path: str | Path) -> None:
        """Save config and model weights to a directory.

        Creates:
            <path>/config.yaml
            <path>/model.pt
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.config.to_yaml(path / "config.yaml")
        torch.save(self.state_dict(), path / "model.pt")
