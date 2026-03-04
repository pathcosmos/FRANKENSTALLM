"""
model — LLM architecture package.

Public API:
    LLM       : top-level decoder-only transformer language model
    LMConfig  : configuration dataclass
"""

from .config import LMConfig
from .transformer import LLM

__all__ = [
    "LLM",
    "LMConfig",
]
