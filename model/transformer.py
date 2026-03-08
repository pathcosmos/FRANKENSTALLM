"""
Full transformer: TransformerBlock and top-level LLM model.
Supports pure Transformer and hybrid Mamba-2 + Transformer architectures.
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
from .mamba_block import Mamba2Block

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
# HuggingFace ↔ Custom weight conversion helpers
# ---------------------------------------------------------------------------

def _load_hf_state_dict(path: Path) -> dict[str, torch.Tensor]:
    """Load weights from HF safetensors (or pytorch_model.bin fallback)."""
    safetensors_path = path / "model.safetensors"
    if safetensors_path.exists():
        from safetensors.torch import load_file
        return load_file(str(safetensors_path), device="cpu")
    bin_path = path / "pytorch_model.bin"
    if bin_path.exists():
        return torch.load(bin_path, map_location="cpu", weights_only=True)
    raise FileNotFoundError(f"No model.safetensors or pytorch_model.bin in {path}")


def _convert_hf_to_custom(hf_sd: dict[str, torch.Tensor], config: LMConfig) -> dict[str, torch.Tensor]:
    """Convert HuggingFace LlamaForCausalLM state dict to our custom format.

    Key mapping:
      HF: model.embed_tokens.weight       → embedding.weight
      HF: model.layers.{i}.self_attn.q/k/v_proj.weight → layers.{i}.attn.qkv_proj.weight (fused)
      HF: model.layers.{i}.self_attn.o_proj.weight     → layers.{i}.attn.out_proj.weight
      HF: model.layers.{i}.input_layernorm.weight      → layers.{i}.attn_norm.weight
      HF: model.layers.{i}.mlp.gate_proj.weight        → layers.{i}.ffn.gate_proj.weight
      HF: model.layers.{i}.mlp.up_proj.weight          → layers.{i}.ffn.up_proj.weight
      HF: model.layers.{i}.mlp.down_proj.weight        → layers.{i}.ffn.down_proj.weight
      HF: model.layers.{i}.post_attention_layernorm.weight → layers.{i}.ffn_norm.weight
      HF: model.norm.weight                → norm.weight
      HF: lm_head.weight                   → lm_head.weight
    """
    sd: dict[str, torch.Tensor] = {}

    sd["embedding.weight"] = hf_sd["model.embed_tokens.weight"]
    sd["norm.weight"] = hf_sd["model.norm.weight"]
    sd["lm_head.weight"] = hf_sd["lm_head.weight"]

    for i in range(config.n_layers):
        pfx = f"model.layers.{i}"
        out = f"layers.{i}"

        # Fuse Q, K, V into single qkv_proj
        q = hf_sd[f"{pfx}.self_attn.q_proj.weight"]
        k = hf_sd[f"{pfx}.self_attn.k_proj.weight"]
        v = hf_sd[f"{pfx}.self_attn.v_proj.weight"]
        sd[f"{out}.attn.qkv_proj.weight"] = torch.cat([q, k, v], dim=0)

        sd[f"{out}.attn.out_proj.weight"] = hf_sd[f"{pfx}.self_attn.o_proj.weight"]
        sd[f"{out}.attn_norm.weight"] = hf_sd[f"{pfx}.input_layernorm.weight"]

        sd[f"{out}.ffn.gate_proj.weight"] = hf_sd[f"{pfx}.mlp.gate_proj.weight"]
        sd[f"{out}.ffn.up_proj.weight"] = hf_sd[f"{pfx}.mlp.up_proj.weight"]
        sd[f"{out}.ffn.down_proj.weight"] = hf_sd[f"{pfx}.mlp.down_proj.weight"]
        sd[f"{out}.ffn_norm.weight"] = hf_sd[f"{pfx}.post_attention_layernorm.weight"]

    return sd


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

        # --- Layers (pure Transformer or hybrid Mamba-Transformer) -----------
        if config.use_hybrid and config.hybrid_pattern:
            pattern = config.hybrid_pattern.strip().split()
            if len(pattern) != config.n_layers:
                raise ValueError(
                    f"hybrid_pattern has {len(pattern)} entries but "
                    f"n_layers={config.n_layers}"
                )
            layers: list[nn.Module] = []
            # Track which layers are Mamba vs Attention for forward dispatch
            self._layer_types: list[str] = pattern
            for layer_type in pattern:
                if layer_type == "M":
                    layers.append(Mamba2Block(
                        d_model=config.d_model,
                        d_state=config.mamba_d_state,
                        head_dim=config.mamba_head_dim,
                        expand=config.mamba_expand,
                        conv_kernel=config.mamba_conv_kernel,
                        n_groups=config.mamba_n_groups,
                        chunk_size=config.mamba_chunk_size,
                    ))
                elif layer_type == "A":
                    layers.append(TransformerBlock(config))
                else:
                    raise ValueError(
                        f"Unknown layer type '{layer_type}' in hybrid_pattern. "
                        f"Use 'M' (Mamba) or 'A' (Attention)."
                    )
            self.layers = nn.ModuleList(layers)
        else:
            self._layer_types = ["A"] * config.n_layers
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
        - Mamba2Block: skipped (manages its own init)
        """
        # TE modules handle their own weight initialisation.
        if HAS_TE and isinstance(module, (te.Linear, te.LayerNormMLP)):
            return
        # Mamba2Block handles its own parameter init (A_log, D, dt_bias, etc.)
        if isinstance(module, Mamba2Block):
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
        # Only needed for Attention layers, but precomputed once for all.
        cos, sin = self.rope(T, device)

        # Run through blocks — Mamba blocks ignore cos/sin
        for layer, ltype in zip(self.layers, self._layer_types):
            if ltype == "M":
                x = layer(x)
            else:
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

        Supports two formats (auto-detected):
          1. Custom: config.yaml + model.pt
          2. HuggingFace: config.json + model.safetensors (LlamaForCausalLM)
        """
        path = Path(path)

        # --- Custom format ---
        if (path / "config.yaml").exists():
            config = LMConfig.from_yaml(path / "config.yaml")
            model = cls(config)
            state_dict = torch.load(
                path / "model.pt",
                map_location="cpu",
                weights_only=True,
            )
            model.load_state_dict(state_dict)
            return model

        # --- HuggingFace format ---
        if (path / "config.json").exists():
            config = LMConfig.from_hf_config(path / "config.json")
            model = cls(config)
            hf_sd = _load_hf_state_dict(path)
            our_sd = _convert_hf_to_custom(hf_sd, config)
            model.load_state_dict(our_sd)
            return model

        raise FileNotFoundError(
            f"No config.yaml or config.json found in {path}"
        )

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
