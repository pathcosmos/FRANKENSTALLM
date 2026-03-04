"""
calibration_task.py — Top-k accuracy and entropy calibration evaluation.

Top-level function for ProcessPoolExecutor (spawn) compatibility:
  - eval_calibration(device, n_tokens=50000) -> dict
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

CHECKPOINT = str(_PROJECT_ROOT / "checkpoints" / "korean_3b_fp8_run1" / "checkpoint-0057000")
TOKENIZER_PATH = str(_PROJECT_ROOT / "tokenizer" / "korean_sp" / "tokenizer.json")
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

def eval_calibration(device: str, n_tokens: int = 50000) -> dict:
    """Compute top-k accuracy and entropy calibration on 3b_val.bin.

    Measures how well the model's probability distribution is calibrated:
      - Top-1/5/10 next-token prediction accuracy
      - Mean probability assigned to the correct next token
      - Mean Shannon entropy of the predictive distribution

    Args:
        device:   CUDA device string, e.g. "cuda:3".
        n_tokens: Number of tokens to evaluate (first n_tokens of 3b_val.bin).

    Returns:
        Dict with keys: n_eval_tokens, top1_accuracy, top5_accuracy,
        top10_accuracy, mean_correct_prob, mean_entropy, elapsed_sec.
    """
    torch.cuda.set_device(int(device.split(":")[-1]))
    print(f"[CALIB {device}] Loading model...")
    model = _load_model(device)

    val_path = DATA_DIR / "3b_val.bin"
    tokens = np.fromfile(str(val_path), dtype=np.uint16)
    tokens = tokens[: min(n_tokens, len(tokens))]
    print(f"[CALIB {device}] Using {len(tokens):,} tokens from 3b_val.bin")

    ds = SlidingWindowDataset(tokens, SEQ_LEN, STRIDE)
    dl = DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    top1_correct = 0
    top5_correct = 0
    top10_correct = 0
    total_entropy = 0.0
    total_prob = 0.0
    total_count = 0
    t0 = time.time()

    with torch.inference_mode():
        for batch_idx, (inp, tgt, mask) in enumerate(dl):
            inp = inp.to(device)
            tgt = tgt.to(device)
            mask = mask.to(device)

            logits, _ = model(inp)
            probs = F.softmax(logits, dim=-1)

            valid = mask & (tgt != -100)
            if valid.sum() == 0:
                continue

            flat_logits = logits[valid]
            flat_tgt = tgt[valid]
            flat_probs = probs[valid]

            # Top-k accuracy
            _, top1_pred = flat_logits.topk(1, dim=-1)
            _, top5_pred = flat_logits.topk(5, dim=-1)
            _, top10_pred = flat_logits.topk(10, dim=-1)

            top1_correct += (top1_pred.squeeze(-1) == flat_tgt).sum().item()
            top5_correct += (
                (top5_pred == flat_tgt.unsqueeze(-1)).any(dim=-1).sum().item()
            )
            top10_correct += (
                (top10_pred == flat_tgt.unsqueeze(-1)).any(dim=-1).sum().item()
            )

            # Mean probability of correct token
            correct_probs = flat_probs[torch.arange(len(flat_tgt), device=device), flat_tgt]
            total_prob += correct_probs.sum().item()

            # Shannon entropy: H = -sum(p * log(p))
            log_probs = torch.log(flat_probs + 1e-10)
            entropy = -(flat_probs * log_probs).sum(dim=-1)
            total_entropy += entropy.sum().item()

            total_count += valid.sum().item()

            if (batch_idx + 1) % 50 == 0:
                elapsed = time.time() - t0
                print(
                    f"[CALIB {device}] batch {batch_idx + 1}/{len(dl)}, "
                    f"tokens so far={total_count:,}, {elapsed:.0f}s"
                )

    elapsed = time.time() - t0
    result: dict = {
        "n_eval_tokens": int(total_count),
        "top1_accuracy": round(top1_correct / total_count, 4) if total_count > 0 else 0.0,
        "top5_accuracy": round(top5_correct / total_count, 4) if total_count > 0 else 0.0,
        "top10_accuracy": round(top10_correct / total_count, 4) if total_count > 0 else 0.0,
        "mean_correct_prob": round(total_prob / total_count, 4) if total_count > 0 else 0.0,
        "mean_entropy": round(total_entropy / total_count, 4) if total_count > 0 else 0.0,
        "elapsed_sec": round(elapsed, 1),
    }
    print(
        f"[CALIB {device}] DONE top1={result['top1_accuracy']:.4f}, "
        f"top5={result['top5_accuracy']:.4f}, "
        f"top10={result['top10_accuracy']:.4f}, "
        f"entropy={result['mean_entropy']:.4f}, {elapsed:.1f}s"
    )
    return result
