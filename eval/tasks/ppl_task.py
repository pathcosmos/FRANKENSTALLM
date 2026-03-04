"""
ppl_task.py — Sliding-window perplexity evaluation task.

Top-level functions for ProcessPoolExecutor (spawn) compatibility:
  - eval_ppl_single(val_file, device, model=None) -> dict
  - eval_ppl_multi(val_files, device) -> list[dict]
"""
from __future__ import annotations

import math
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
    """Sliding-window tokenized dataset for perplexity evaluation."""

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
# Main task functions (must be top-level for pickle / spawn compatibility)
# ---------------------------------------------------------------------------

def eval_ppl_single(val_file: str, device: str, model=None) -> dict:
    """Compute sliding-window perplexity for a single validation file.

    Args:
        val_file: Relative path under DATA_DIR, e.g. "3b_val.bin".
        device:   CUDA device string, e.g. "cuda:0".
        model:    Optional pre-loaded model. If None, loads from checkpoint.

    Returns:
        Dict with keys: name, file, n_tokens, n_eval_tokens, ppl,
        bits_per_token, avg_nll, elapsed_sec, device.
    """
    torch.cuda.set_device(int(device.split(":")[-1]))
    data_path = DATA_DIR / val_file
    name = val_file.replace("_val.bin", "").replace(".bin", "")

    own_model = model is None
    if own_model:
        print(f"[PPL {device}] Loading model for {name}...")
        model = _load_model(device)

    tokens = np.fromfile(str(data_path), dtype=np.uint16)
    n_tokens = len(tokens)
    print(f"[PPL {device}] {name}: {n_tokens:,} tokens, {n_tokens * 2 / 1e6:.1f} MB")

    ds = SlidingWindowDataset(tokens, SEQ_LEN, STRIDE)
    dl = DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    total_nll = 0.0
    total_count = 0
    t0 = time.time()

    with torch.inference_mode():
        for batch_idx, (inp, tgt, mask) in enumerate(dl):
            inp = inp.to(device)
            tgt = tgt.to(device)
            mask = mask.to(device)

            logits, _ = model(inp)
            loss_flat = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                tgt.view(-1),
                reduction="none",
            )
            loss_flat = loss_flat.view(mask.shape)
            nll = (loss_flat * mask.float()).sum().item()
            cnt = mask.sum().item()
            total_nll += nll
            total_count += cnt

            if (batch_idx + 1) % 50 == 0:
                running_ppl = (
                    math.exp(total_nll / total_count) if total_count > 0 else float("inf")
                )
                elapsed = time.time() - t0
                print(
                    f"[PPL {device}] {name}: batch {batch_idx + 1}/{len(dl)}, "
                    f"running PPL={running_ppl:.4f}, {elapsed:.0f}s"
                )

    avg_nll = total_nll / total_count if total_count > 0 else 0.0
    ppl = math.exp(avg_nll)
    bpt = avg_nll / math.log(2)
    elapsed = time.time() - t0

    result: dict = {
        "name": name,
        "file": val_file,
        "n_tokens": int(n_tokens),
        "n_eval_tokens": int(total_count),
        "ppl": round(ppl, 4),
        "bits_per_token": round(bpt, 4),
        "avg_nll": round(avg_nll, 6),
        "elapsed_sec": round(elapsed, 1),
        "device": device,
    }
    print(
        f"[PPL {device}] DONE {name}: PPL={ppl:.4f}, BPT={bpt:.4f}, {elapsed:.1f}s"
    )
    return result


def eval_ppl_multi(val_files: list[str], device: str) -> list[dict]:
    """Compute PPL for multiple val files on a single GPU, loading model once.

    Args:
        val_files: List of relative paths under DATA_DIR.
        device:    CUDA device string.

    Returns:
        List of result dicts (one per file), in the same order as val_files.
    """
    torch.cuda.set_device(int(device.split(":")[-1]))
    print(f"[PPL_MULTI {device}] Loading model once for {len(val_files)} files...")
    model = _load_model(device)

    results: list[dict] = []
    for val_file in val_files:
        result = eval_ppl_single(val_file, device, model=model)
        results.append(result)

    return results
