"""
token_nll_task.py — Token-level NLL distribution analysis.

Top-level function for ProcessPoolExecutor (spawn) compatibility:
  - eval_token_nll(device, n_tokens=50000) -> dict
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

_DEFAULT_CHECKPOINT = str(_PROJECT_ROOT / "checkpoints" / "korean_3b_fp8_run1" / "checkpoint-0057000")
CHECKPOINT = os.environ.get("EVAL_CHECKPOINT", _DEFAULT_CHECKPOINT)
TOKENIZER_PATH = os.environ.get("EVAL_TOKENIZER", str(_PROJECT_ROOT / "tokenizer" / "korean_sp" / "tokenizer.json"))
DATA_DIR = _PROJECT_ROOT / "data"
SEQ_LEN = 2048
STRIDE = 512
BATCH_SIZE = 32


# ---------------------------------------------------------------------------
# Shared dataset / model utilities
# ---------------------------------------------------------------------------

class SlidingWindowDataset(Dataset):
    """Sliding-window tokenized dataset for evaluation."""

    def __init__(self, tokens: np.ndarray, seq_len: int, stride: int) -> None:
        self.tokens = tokens
        self.seq_len = seq_len
        self.stride = stride
        self.n_windows = max(0, (len(tokens) - seq_len + stride - 1) // stride)

    def __len__(self) -> int:
        return self.n_windows

    def __getitem__(self, idx: int):
        start = idx * self.stride
        end = start + self.seq_len
        actual_end = min(end, len(self.tokens))
        chunk_len = actual_end - start

        input_ids = torch.zeros(self.seq_len, dtype=torch.long)
        targets = torch.full((self.seq_len,), fill_value=-100, dtype=torch.long)
        loss_mask = torch.zeros(self.seq_len, dtype=torch.bool)

        if chunk_len > 1:
            toks = torch.from_numpy(self.tokens[start:actual_end].astype(np.int64))
            input_ids[:chunk_len] = toks
            targets[:chunk_len - 1] = toks[1:]

        new_start = 0 if idx == 0 else self.stride
        if chunk_len > 1:
            for pos in range(new_start, chunk_len - 1):
                loss_mask[pos] = True

        return input_ids, targets, loss_mask


def _load_model(device: str):
    """Load FRANKENSTALLM 3B from checkpoint onto the given device."""
    from model.transformer import LLM  # type: ignore[import]

    model = LLM.from_pretrained(CHECKPOINT)
    model = model.to(device=device, dtype=torch.bfloat16)
    model.eval()
    return model


def _load_tokenizer():
    """Load the Korean SentencePiece tokenizer."""
    from tokenizers import Tokenizer  # type: ignore[import]

    return Tokenizer.from_file(TOKENIZER_PATH)


# ---------------------------------------------------------------------------
# Main task function (must be top-level for pickle / spawn compatibility)
# ---------------------------------------------------------------------------

def eval_token_nll(device: str, n_tokens: int = 50000) -> dict:
    """Analyse the per-token NLL distribution on 3b_val.bin.

    Collects the NLL of every valid (unmasked) token and computes summary
    statistics and percentile breakdowns, as well as the fraction of
    "high-loss" tokens that may indicate out-of-distribution content.

    Args:
        device:   CUDA device string, e.g. "cuda:6".
        n_tokens: Number of tokens to process (first n_tokens of 3b_val.bin).

    Returns:
        Dict with keys:
          - n_eval_tokens: number of tokens included in stats
          - nll_mean:      mean token NLL
          - nll_std:       standard deviation of token NLL
          - nll_median:    50th-percentile NLL
          - nll_percentiles: dict mapping percentile label to value
            (keys: p5, p25, p75, p95, p99)
          - high_loss_fraction_5:  fraction of tokens with NLL > 5.0
          - high_loss_fraction_10: fraction of tokens with NLL > 10.0
          - elapsed_sec:  wall-clock time
    """
    torch.cuda.set_device(int(device.split(":")[-1]))
    print(f"[NLL {device}] Loading model...")
    model = _load_model(device)

    val_path = DATA_DIR / "3b_val.bin"
    if not val_path.exists():
        raise FileNotFoundError(f"Validation file not found: {val_path}")
    tokens = np.fromfile(str(val_path), dtype=np.uint16)
    if len(tokens) == 0:
        raise ValueError(f"Validation file is empty (0 tokens): {val_path}")
    tokens = tokens[: min(n_tokens, len(tokens))]
    print(f"[NLL {device}] Using {len(tokens):,} tokens from 3b_val.bin")

    ds = SlidingWindowDataset(tokens, SEQ_LEN, STRIDE)
    dl = DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    all_nlls: list[np.ndarray] = []
    t0 = time.time()

    with torch.inference_mode():
        for batch_idx, (inp, tgt, mask) in enumerate(dl):
            inp = inp.to(device)
            tgt = tgt.to(device)
            mask = mask.to(device)

            logits, _ = model(inp)

            # Per-token NLL — shape (batch, seq_len)
            per_token_nll = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                tgt.view(-1),
                reduction="none",
                ignore_index=-100,
            ).view(mask.shape)

            # Collect only valid (unmasked) positions
            valid_nll = per_token_nll[mask].float().cpu().numpy()
            if len(valid_nll) > 0:
                all_nlls.append(valid_nll)

            if (batch_idx + 1) % 50 == 0:
                n_collected = sum(len(a) for a in all_nlls)
                elapsed = time.time() - t0
                print(
                    f"[NLL {device}] batch {batch_idx + 1}/{len(dl)}, "
                    f"tokens collected={n_collected:,}, {elapsed:.0f}s"
                )

    elapsed = time.time() - t0

    if all_nlls:
        nll_arr = np.concatenate(all_nlls)
    else:
        nll_arr = np.array([], dtype=np.float32)

    n_eval = len(nll_arr)

    if n_eval > 0:
        nll_mean = float(np.mean(nll_arr))
        nll_std = float(np.std(nll_arr))
        nll_median = float(np.median(nll_arr))
        percentiles = {
            "p5":  round(float(np.percentile(nll_arr, 5)),  4),
            "p25": round(float(np.percentile(nll_arr, 25)), 4),
            "p75": round(float(np.percentile(nll_arr, 75)), 4),
            "p95": round(float(np.percentile(nll_arr, 95)), 4),
            "p99": round(float(np.percentile(nll_arr, 99)), 4),
        }
        high_loss_5  = float(np.mean(nll_arr > 5.0))
        high_loss_10 = float(np.mean(nll_arr > 10.0))
    else:
        nll_mean = nll_std = nll_median = 0.0
        percentiles = {"p5": 0.0, "p25": 0.0, "p75": 0.0, "p95": 0.0, "p99": 0.0}
        high_loss_5 = high_loss_10 = 0.0

    result: dict = {
        "n_eval_tokens": int(n_eval),
        "nll_mean": round(nll_mean, 4),
        "nll_std": round(nll_std, 4),
        "nll_median": round(nll_median, 4),
        "nll_percentiles": {k: round(v, 4) for k, v in percentiles.items()},
        "high_loss_fraction_5": round(high_loss_5, 6),
        "high_loss_fraction_10": round(high_loss_10, 6),
        "elapsed_sec": round(elapsed, 1),
    }

    print(
        f"[NLL {device}] DONE n={n_eval:,}, "
        f"mean={nll_mean:.4f}, std={nll_std:.4f}, "
        f"median={nll_median:.4f}, "
        f"high_loss(>5)={high_loss_5:.2%}, "
        f"high_loss(>10)={high_loss_10:.2%}, "
        f"{elapsed:.1f}s"
    )
    return result
