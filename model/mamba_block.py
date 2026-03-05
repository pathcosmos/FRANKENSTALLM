"""
Mamba-2 block based on the Structured State Space Duality (SSD) formulation.

Reference: "Transformers are SSMs: Generalized Models and Efficient Algorithms
Through Structured State Space Duality" (Dao & Gu, 2024).

This implements a pure-PyTorch sequential scan for correctness and generality.
A chunked SSD kernel can be swapped in later for speed.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import RMSNorm


# ---------------------------------------------------------------------------
# Selective Scan (sequential, numerically stable in float32)
# ---------------------------------------------------------------------------

def selective_scan(
    x: torch.Tensor,
    dt: torch.Tensor,
    A_log: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor,
    n_groups: int,
) -> torch.Tensor:
    """Run the SSM recurrence sequentially over the time axis.

    Args:
        x:     (B, L, n_heads, head_dim) — input after conv + activation.
        dt:    (B, L, n_heads)           — discretisation time-steps (after softplus).
        A_log: (n_heads,)                — log(-A), learnable diagonal decay.
        B:     (B, L, n_groups, d_state) — input-to-state projection per step.
        C:     (B, L, n_groups, d_state) — state-to-output projection per step.
        D:     (n_heads,)                — skip/residual connection per head.
        n_groups: int                    — number of B/C groups (heads per group share B/C).

    Returns:
        y: (B, L, n_heads, head_dim) — SSM output.
    """
    batch, seq_len, n_heads, head_dim = x.shape
    d_state = B.shape[-1]
    heads_per_group = n_heads // n_groups

    # Compute decay: dA = exp(-exp(A_log) * dt)  — shape (B, L, n_heads)
    neg_A = A_log.exp()                           # (n_heads,)
    dA = torch.exp(-neg_A.unsqueeze(0).unsqueeze(0) * dt)  # (B, L, n_heads)

    # Scale input by dt: dBx will be accumulated into state
    # dt: (B, L, n_heads) -> (B, L, n_heads, 1)
    dt_x = dt.unsqueeze(-1) * x  # (B, L, n_heads, head_dim)

    # Allocate output
    y = torch.zeros_like(x)

    # State: (B, n_heads, head_dim, d_state) — accumulated in float32
    h = torch.zeros(
        batch, n_heads, head_dim, d_state,
        dtype=torch.float32, device=x.device,
    )

    # Expand B/C from groups to heads: (B, L, n_groups, d_state) -> indexing
    # For efficiency we index into the group dimension during the loop.
    # group_idx[head] -> which group this head belongs to
    group_idx = torch.arange(n_heads, device=x.device) // heads_per_group  # (n_heads,)

    for t in range(seq_len):
        # --- Decay state ---
        # dA_t: (B, n_heads) -> (B, n_heads, 1, 1)
        dA_t = dA[:, t, :].float().unsqueeze(-1).unsqueeze(-1)
        h = h * dA_t  # (B, n_heads, head_dim, d_state)

        # --- Input contribution ---
        # B_t: (B, n_groups, d_state) -> (B, n_heads, d_state) via group expansion
        B_t = B[:, t, :, :][:, group_idx, :]  # (B, n_heads, d_state)
        # dt_x_t: (B, n_heads, head_dim)
        dt_x_t = dt_x[:, t, :, :].float()     # (B, n_heads, head_dim)
        # Outer product: (B, n_heads, head_dim, 1) * (B, n_heads, 1, d_state)
        h = h + dt_x_t.unsqueeze(-1) * B_t.float().unsqueeze(-2)

        # --- Output ---
        # C_t: (B, n_groups, d_state) -> (B, n_heads, d_state)
        C_t = C[:, t, :, :][:, group_idx, :]  # (B, n_heads, d_state)
        # y_t = sum_over_d_state( h * C_t ) -> (B, n_heads, head_dim)
        y_t = torch.einsum("bnhd,bnd->bnh", h, C_t.float())
        y[:, t, :, :] = y_t.to(x.dtype)

    # Skip connection: D * x
    y = y + D.view(1, 1, n_heads, 1) * x

    return y


# ---------------------------------------------------------------------------
# Mamba-2 Block
# ---------------------------------------------------------------------------

class Mamba2Block(nn.Module):
    """Mamba-2 block with pre-norm residual connection.

    Implements:
        1. RMSNorm (pre-norm)
        2. Input projection -> (z, x, B, C, dt)
        3. Causal depth-wise Conv1d on x
        4. SiLU activation on x
        5. Selective scan (SSM recurrence)
        6. Gated output: y * SiLU(z)
        7. Output projection + residual

    Args:
        d_model:     Model hidden dimension.
        d_state:     SSM state dimension N (default 128).
        head_dim:    Per-head dimension for SSD (default 64).
        expand:      Expansion factor for inner dimension (default 2).
        conv_kernel: Causal 1D convolution kernel size (default 4).
        n_groups:    Number of groups for B/C projections (default 1).
        chunk_size:  Chunk size for SSD algorithm — reserved for future use (default 256).
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 128,
        head_dim: int = 64,
        expand: int = 2,
        conv_kernel: int = 4,
        n_groups: int = 1,
        chunk_size: int = 256,
    ) -> None:
        super().__init__()

        self.d_model = d_model
        self.d_state = d_state
        self.head_dim = head_dim
        self.expand = expand
        self.n_groups = n_groups
        self.chunk_size = chunk_size

        # Derived dimensions
        self.d_inner = expand * d_model
        self.n_heads = self.d_inner // head_dim
        assert self.d_inner % head_dim == 0, (
            f"d_inner ({self.d_inner}) must be divisible by head_dim ({head_dim})"
        )
        assert self.n_heads % n_groups == 0, (
            f"n_heads ({self.n_heads}) must be divisible by n_groups ({n_groups})"
        )

        # Pre-norm
        self.norm = RMSNorm(d_model)

        # Input projection: d_model -> z + x + B + C + dt
        self.d_proj = (
            self.d_inner          # z (gate)
            + self.d_inner        # x (input to conv + SSM)
            + n_groups * d_state  # B
            + n_groups * d_state  # C
            + self.n_heads        # dt (one per head)
        )
        self.in_proj = nn.Linear(d_model, self.d_proj, bias=False)

        # Causal depth-wise conv1d over x
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=conv_kernel,
            groups=self.d_inner,
            padding=conv_kernel - 1,  # causal: trim trailing values
        )

        # SSM parameters
        # A_log: log(-A) where A is the diagonal decay — init from log(uniform(1, 16))
        A_init = torch.log(torch.rand(self.n_heads) * 15.0 + 1.0)  # log(U(1,16))
        self.A_log = nn.Parameter(A_init)

        # D: skip connection per head — init to ones
        self.D = nn.Parameter(torch.ones(self.n_heads))

        # dt_bias: added before softplus — init from log(uniform(0.001, 0.1))
        dt_bias_init = torch.log(torch.rand(self.n_heads) * 0.099 + 0.001)
        self.dt_bias = nn.Parameter(dt_bias_init)

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _split_projection(
        self, proj: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Split the fused input projection into (z, x, B, C, dt).

        Args:
            proj: (B, L, d_proj)

        Returns:
            z:  (B, L, d_inner)
            x:  (B, L, d_inner)
            B:  (B, L, n_groups, d_state)
            C:  (B, L, n_groups, d_state)
            dt: (B, L, n_heads)
        """
        batch, seq_len, _ = proj.shape
        i = 0

        z = proj[:, :, i : i + self.d_inner]
        i += self.d_inner

        x = proj[:, :, i : i + self.d_inner]
        i += self.d_inner

        bc_dim = self.n_groups * self.d_state
        B = proj[:, :, i : i + bc_dim].reshape(batch, seq_len, self.n_groups, self.d_state)
        i += bc_dim

        C = proj[:, :, i : i + bc_dim].reshape(batch, seq_len, self.n_groups, self.d_state)
        i += bc_dim

        dt = proj[:, :, i : i + self.n_heads]
        return z, x, B, C, dt

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, L, d_model) — input hidden states.

        Returns:
            (B, L, d_model) — output with residual connection applied.
        """
        residual = x
        x = self.norm(x)

        # --- Input projection ---
        proj = self.in_proj(x)                         # (B, L, d_proj)
        z, x_ssm, B, C, dt_raw = self._split_projection(proj)

        # --- Causal conv1d on x ---
        # Conv1d expects (B, C, L)
        x_conv = x_ssm.transpose(1, 2)                # (B, d_inner, L)
        x_conv = self.conv1d(x_conv)
        # Trim to causal: remove the (kernel-1) trailing padding
        x_conv = x_conv[:, :, :x_ssm.shape[1]]        # (B, d_inner, L)
        x_conv = x_conv.transpose(1, 2)               # (B, L, d_inner)
        x_conv = F.silu(x_conv)

        # --- Discretise dt ---
        dt = F.softplus(dt_raw + self.dt_bias)         # (B, L, n_heads)

        # --- Reshape x for multi-head scan ---
        batch, seq_len, _ = x_conv.shape
        x_heads = x_conv.reshape(batch, seq_len, self.n_heads, self.head_dim)

        # --- Selective scan (SSM recurrence) ---
        y = selective_scan(
            x_heads, dt, self.A_log, B, C, self.D,
            n_groups=self.n_groups,
        )  # (B, L, n_heads, head_dim)

        # --- Flatten heads back ---
        y = y.reshape(batch, seq_len, self.d_inner)    # (B, L, d_inner)

        # --- Gated output ---
        y = y * F.silu(z)

        # --- Output projection + residual ---
        return residual + self.out_proj(y)
