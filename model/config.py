"""
LMConfig: configuration dataclass for the LLM model architecture.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import json

import yaml


def _round_to_multiple(n: int, multiple: int) -> int:
    """Round n up to the nearest multiple of `multiple`."""
    return math.ceil(n / multiple) * multiple


@dataclass
class LMConfig:
    # Vocabulary
    vocab_size: int = 32000

    # Model dimensions
    d_model: int = 768
    n_layers: int = 12
    n_heads: int = 12

    # Grouped-query attention: None → standard MHA (n_kv_heads == n_heads)
    n_kv_heads: Optional[int] = None

    # Feed-forward hidden dimension: None → auto-computed
    d_ffn: Optional[int] = None

    # Sequence length
    max_seq_len: int = 2048

    # RoPE base frequency
    rope_theta: float = 10000.0

    # Regularisation
    dropout: float = 0.0
    bias: bool = False

    # Attention backend
    use_flash_attn: bool = True

    # FP8 quantization
    use_fp8: bool = False

    # Hybrid Mamba-Transformer settings
    use_hybrid: bool = False
    hybrid_pattern: str = ""  # e.g. "M M A M M M M A M M M M M M M M M M A M" for 40-layer Nemotron-H style
    # Mamba-2 SSM parameters
    mamba_d_state: int = 128
    mamba_head_dim: int = 64
    mamba_expand: int = 2
    mamba_conv_kernel: int = 4
    mamba_n_groups: int = 1
    mamba_chunk_size: int = 256

    def __post_init__(self) -> None:
        # Resolve n_kv_heads: None → full MHA
        if self.n_kv_heads is None:
            self.n_kv_heads = self.n_heads

        # Validate GQA divisibility
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError(
                f"n_heads ({self.n_heads}) must be divisible by "
                f"n_kv_heads ({self.n_kv_heads})"
            )

        # Compute d_ffn using the LLaMA-style formula: round(8/3 * d_model)
        # rounded up to the nearest multiple of 256.
        if self.d_ffn is None:
            raw = int(8 / 3 * self.d_model)
            self.d_ffn = _round_to_multiple(raw, 256)

        # Hybrid Mamba-Transformer validation
        if self.use_hybrid and not self.hybrid_pattern.strip():
            raise ValueError(
                "use_hybrid=True requires a non-empty hybrid_pattern "
                "(space-separated 'M'/'A' per layer)"
            )

        # FP8 alignment: TE requires dimensions divisible by 16
        if self.use_fp8:
            if self.d_model % 16 != 0:
                raise ValueError(f"FP8: d_model ({self.d_model}) must be divisible by 16")
            if self.d_ffn % 16 != 0:
                raise ValueError(f"FP8: d_ffn ({self.d_ffn}) must be divisible by 16")

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def num_params(self) -> int:
        """Approximate parameter count using the 12 * L * d^2 rule."""
        return 12 * self.n_layers * self.d_model ** 2

    @property
    def head_dim(self) -> int:
        """Dimensionality of each attention head."""
        return self.d_model // self.n_heads

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Return a plain-Python-dict representation of the config."""
        return {
            "vocab_size": self.vocab_size,
            "d_model": self.d_model,
            "n_layers": self.n_layers,
            "n_heads": self.n_heads,
            "n_kv_heads": self.n_kv_heads,
            "d_ffn": self.d_ffn,
            "max_seq_len": self.max_seq_len,
            "rope_theta": self.rope_theta,
            "dropout": self.dropout,
            "bias": self.bias,
            "use_flash_attn": self.use_flash_attn,
            "use_fp8": self.use_fp8,
            "use_hybrid": self.use_hybrid,
            "hybrid_pattern": self.hybrid_pattern,
            "mamba_d_state": self.mamba_d_state,
            "mamba_head_dim": self.mamba_head_dim,
            "mamba_expand": self.mamba_expand,
            "mamba_conv_kernel": self.mamba_conv_kernel,
            "mamba_n_groups": self.mamba_n_groups,
            "mamba_chunk_size": self.mamba_chunk_size,
        }

    def to_yaml(self, path: str | Path) -> None:
        """Serialise config to a YAML file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

    @classmethod
    def from_dict(cls, d: dict) -> "LMConfig":
        """Construct a LMConfig from a plain dict (e.g. loaded from YAML)."""
        return cls(**d)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "LMConfig":
        """Load config from a YAML file."""
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        # Support nested YAML with 'model' section (e.g., shared multi-section configs)
        if "model" in data and isinstance(data["model"], dict):
            data = data["model"]
        return cls.from_dict(data)

    @classmethod
    def from_hf_config(cls, path: str | Path) -> "LMConfig":
        """Load config from a HuggingFace-format config.json (LlamaForCausalLM)."""
        path = Path(path)
        with open(path, "r", encoding="utf-8") as f:
            hf = json.load(f)

        rope_theta = 10000.0
        if "rope_parameters" in hf and isinstance(hf["rope_parameters"], dict):
            rope_theta = float(hf["rope_parameters"].get("rope_theta", rope_theta))
        elif "rope_theta" in hf:
            rope_theta = float(hf["rope_theta"])

        return cls(
            vocab_size=hf["vocab_size"],
            d_model=hf["hidden_size"],
            n_layers=hf["num_hidden_layers"],
            n_heads=hf["num_attention_heads"],
            n_kv_heads=hf.get("num_key_value_heads", hf["num_attention_heads"]),
            d_ffn=hf["intermediate_size"],
            max_seq_len=hf.get("max_position_embeddings", 4096),
            rope_theta=rope_theta,
            dropout=hf.get("attention_dropout", 0.0),
            bias=hf.get("attention_bias", False),
        )
