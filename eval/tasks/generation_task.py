"""
generation_task.py — Text generation quality evaluation tasks.

Top-level functions for ProcessPoolExecutor (spawn) compatibility:
  - eval_generation(device) -> dict
  - eval_repetition_grid(device) -> dict

Helper functions (also top-level, used internally):
  - top_p_filtering(logits, top_p, top_k)
  - generate_one(model, tokenizer, prompt, temperature, ...)
  - compute_ngram_rep(text, n)
"""
from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

_DEFAULT_CHECKPOINT = str(_PROJECT_ROOT / "checkpoints" / "korean_3b_fp8_run1" / "checkpoint-0057000")
CHECKPOINT = os.environ.get("EVAL_CHECKPOINT", _DEFAULT_CHECKPOINT)
TOKENIZER_PATH = os.environ.get("EVAL_TOKENIZER", str(_PROJECT_ROOT / "tokenizer" / "korean_sp" / "tokenizer.json"))

# Chat template support for SFT models
USE_CHAT_TEMPLATE = os.environ.get("USE_CHAT_TEMPLATE", "0") == "1"
CHAT_TEMPLATE_FMT = "<|user|>\n{prompt}\n<|assistant|>\n"
DATA_DIR = _PROJECT_ROOT / "data"
SEQ_LEN = 2048
STRIDE = 512
BATCH_SIZE = 32

# ---------------------------------------------------------------------------
# Prompt / temperature constants
# ---------------------------------------------------------------------------

PROMPTS = [
    "대한민국의 수도는",
    "인공지능이란",
    "한국의 전통 음식 중에서",
    "지구 온난화의 주요 원인은",
    "프로그래밍을 배우려면",
    "조선시대에는",
    "물리학에서 에너지란",
    "한국어는 세계에서",
    "경제 성장을 위해서는",
    "우주 탐사의 역사를 보면",
    "머신러닝과 딥러닝의 차이는",
    "한국 문학의 대표적인 작품으로는",
    "양자 컴퓨터란",
    "건강한 식습관을 위해서는",
    "세계 2차 대전 이후",
]

TEMPERATURES = [0.0, 0.5, 0.8, 1.0]

REP_GRID = [
    {"name": "greedy",       "temperature": 0.0, "repetition_penalty": 1.0},
    {"name": "t0.5",         "temperature": 0.5, "repetition_penalty": 1.0},
    {"name": "t0.5_rep1.1",  "temperature": 0.5, "repetition_penalty": 1.1},
    {"name": "t0.7",         "temperature": 0.7, "repetition_penalty": 1.0},
    {"name": "t0.7_rep1.1",  "temperature": 0.7, "repetition_penalty": 1.1},
    {"name": "t0.7_rep1.2",  "temperature": 0.7, "repetition_penalty": 1.2},
    {"name": "t0.7_rep1.3",  "temperature": 0.7, "repetition_penalty": 1.3},
    {"name": "t0.9",         "temperature": 0.9, "repetition_penalty": 1.0},
    {"name": "t0.9_rep1.1",  "temperature": 0.9, "repetition_penalty": 1.1},
    {"name": "t0.9_rep1.2",  "temperature": 0.9, "repetition_penalty": 1.2},
    {"name": "t1.0",         "temperature": 1.0, "repetition_penalty": 1.0},
    {"name": "t1.0_rep1.1",  "temperature": 1.0, "repetition_penalty": 1.1},
]


# ---------------------------------------------------------------------------
# Shared model utilities
# ---------------------------------------------------------------------------

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
# Generation helpers (top-level for pickle compatibility)
# ---------------------------------------------------------------------------

def top_p_filtering(logits: torch.Tensor, top_p: float = 0.9, top_k: int = 0) -> torch.Tensor:
    """Apply top-p (nucleus) and/or top-k filtering to a logits tensor.

    Args:
        logits: Shape (..., vocab_size).
        top_p:  Nucleus probability threshold in (0, 1). 0 or 1 disables.
        top_k:  Keep only the top-k tokens. 0 disables.

    Returns:
        Filtered logits tensor of the same shape.
    """
    if logits.dim() == 1:
        logits = logits.unsqueeze(0)
        squeeze = True
    else:
        squeeze = False

    if top_k > 0:
        k = min(top_k, logits.size(-1))
        kth = torch.topk(logits, k, dim=-1).values[:, -1, None]
        logits = logits.masked_fill(logits < kth, float("-inf"))

    if 0.0 < top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, dim=-1, descending=True)
        cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        remove = cum_probs - F.softmax(sorted_logits, dim=-1) >= top_p
        sorted_logits[remove] = float("-inf")
        logits = torch.zeros_like(logits).scatter_(-1, sorted_idx, sorted_logits)

    if squeeze:
        logits = logits.squeeze(0)
    return logits


def generate_one(
    model,
    tokenizer,
    prompt: str,
    temperature: float,
    top_p: float = 0.9,
    top_k: int = 50,
    max_new_tokens: int = 256,
    device: str = "cuda:0",
    repetition_penalty: float = 1.0,
) -> tuple[str, int, bool]:
    """Generate a single continuation for a prompt using the given model.

    Args:
        model:              Pre-loaded language model (eval mode).
        tokenizer:          Tokenizer with encode/decode methods.
        prompt:             Input prompt string.
        temperature:        Sampling temperature. 0.0 = greedy.
        top_p:              Nucleus filtering threshold.
        top_k:              Top-k filtering count.
        max_new_tokens:     Maximum number of tokens to generate.
        device:             CUDA device string.
        repetition_penalty: Penalty > 1.0 discourages token repetition.

    Returns:
        Tuple of (generated_text, num_new_tokens, hit_eos).
    """
    input_ids = torch.tensor(
        [tokenizer.encode(prompt).ids], dtype=torch.long, device=device
    )
    eos_id = tokenizer.token_to_id("</s>")
    generated = input_ids
    new_ids: list[int] = []
    hit_eos = False

    for _ in range(max_new_tokens):
        logits_all, _ = model(generated)
        logits = logits_all[:, -1, :].clone()

        if repetition_penalty != 1.0:
            for tid in set(generated[0].tolist()):
                if logits[0, tid] > 0:
                    logits[0, tid] /= repetition_penalty
                else:
                    logits[0, tid] *= repetition_penalty

        if temperature == 0.0:
            next_id = logits.argmax(dim=-1, keepdim=True)
        else:
            logits = logits / max(temperature, 1e-8)
            logits = top_p_filtering(logits, top_p=top_p, top_k=top_k)
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)

        generated = torch.cat([generated, next_id], dim=-1)
        new_ids.append(next_id.item())

        if eos_id is not None and next_id.item() == eos_id:
            hit_eos = True
            break

    text = tokenizer.decode(new_ids)
    return text, len(new_ids), hit_eos


def compute_ngram_rep(text: str, n: int) -> float:
    """Compute n-gram repetition rate for a whitespace-tokenized string.

    Repetition rate = 1 - (unique n-grams / total n-grams).
    A value of 0 means no repeated n-grams; 1 means all n-grams are repeated.

    Args:
        text: Input text (whitespace-tokenized).
        n:    N-gram order (1, 2, 3, 4, ...).

    Returns:
        Float in [0, 1].
    """
    tokens = text.split()
    if len(tokens) < n:
        return 0.0
    ngrams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
    if not ngrams:
        return 0.0
    return 1.0 - len(set(ngrams)) / len(ngrams)


def compute_diversity_metrics(text: str) -> dict:
    """N-gram 반복률을 보완하는 어휘 다양성 메트릭.

    - Distinct-n (Li et al., 2016): 고유 n-gram 비율
    - Type-Token Ratio: 어휘 풍부도
    """
    tokens = text.split()
    n = len(tokens)
    if n == 0:
        return {"distinct_1": 0.0, "distinct_2": 0.0, "distinct_3": 0.0,
                "type_token_ratio": 0.0, "vocab_size": 0, "total_tokens": 0}

    unigrams = set(tokens)
    bigrams = set(zip(tokens, tokens[1:])) if n > 1 else set()
    trigrams = set(zip(tokens, tokens[1:], tokens[2:])) if n > 2 else set()

    return {
        "distinct_1": len(unigrams) / n,
        "distinct_2": len(bigrams) / max(n - 1, 1),
        "distinct_3": len(trigrams) / max(n - 2, 1),
        "type_token_ratio": len(unigrams) / n,
        "vocab_size": len(unigrams),
        "total_tokens": n,
    }


# ---------------------------------------------------------------------------
# Main task functions (must be top-level for pickle / spawn compatibility)
# ---------------------------------------------------------------------------

def eval_generation(device: str) -> dict:
    """Evaluate generation quality: 15 prompts x 4 temperatures.

    For each (prompt, temperature) combination:
      - Generates up to 256 new tokens
      - Computes 1-gram through 4-gram repetition rates

    Args:
        device: CUDA device string, e.g. "cuda:4".

    Returns:
        Dict with keys:
          - summary: aggregate statistics across all generations
          - samples: list of per-generation result dicts
    """
    torch.cuda.set_device(int(device.split(":")[-1]))
    print(f"[GEN {device}] Loading model...")
    model = _load_model(device)
    tokenizer = _load_tokenizer()
    t0 = time.time()

    results: list[dict] = []
    total_combinations = len(PROMPTS) * len(TEMPERATURES)
    done = 0

    if USE_CHAT_TEMPLATE:
        print(f"[GEN {device}] Chat template ENABLED", flush=True)

    for prompt in PROMPTS:
        effective_prompt = CHAT_TEMPLATE_FMT.format(prompt=prompt) if USE_CHAT_TEMPLATE else prompt
        for temp in TEMPERATURES:
            with torch.inference_mode():
                text, n_tokens, hit_eos = generate_one(
                    model, tokenizer, effective_prompt, temp, device=device
                )
            rep1 = compute_ngram_rep(text, 1)
            rep2 = compute_ngram_rep(text, 2)
            rep3 = compute_ngram_rep(text, 3)
            rep4 = compute_ngram_rep(text, 4)
            diversity = compute_diversity_metrics(text)

            entry = {
                "prompt": prompt,
                "chat_template": USE_CHAT_TEMPLATE,
                "effective_prompt": effective_prompt if USE_CHAT_TEMPLATE else prompt,
                "temperature": temp,
                "generated_tokens": n_tokens,
                "hit_eos": hit_eos,
                "1gram_rep": round(rep1, 4),
                "2gram_rep": round(rep2, 4),
                "3gram_rep": round(rep3, 4),
                "4gram_rep": round(rep4, 4),
                "distinct_1": round(diversity["distinct_1"], 4),
                "distinct_2": round(diversity["distinct_2"], 4),
                "distinct_3": round(diversity["distinct_3"], 4),
                "type_token_ratio": round(diversity["type_token_ratio"], 4),
                "text": text[:500],  # truncate for readability
            }
            results.append(entry)
            done += 1

            label = "greedy" if temp == 0.0 else f"t={temp}"
            print(
                f"[GEN {device}] ({done}/{total_combinations}) "
                f"{prompt[:15]}... ({label}): "
                f"{n_tokens}tok, 3gram_rep={rep3:.2%}, eos={hit_eos}"
            )

    elapsed = time.time() - t0

    # Aggregate stats per temperature group
    greedy = [r for r in results if r["temperature"] == 0.0]
    sampled = [r for r in results if r["temperature"] > 0.0]

    if not greedy:
        logger.warning("No greedy generation results — all prompts may have failed")
    if not sampled:
        logger.warning("No sampled generation results")

    summary = {
        "total_generations": len(results),
        "n_prompts": len(PROMPTS),
        "temperatures": TEMPERATURES,
        "greedy_avg_1gram_rep": round(np.mean([r["1gram_rep"] for r in greedy]), 4) if greedy else 0.0,
        "greedy_avg_2gram_rep": round(np.mean([r["2gram_rep"] for r in greedy]), 4) if greedy else 0.0,
        "greedy_avg_3gram_rep": round(np.mean([r["3gram_rep"] for r in greedy]), 4) if greedy else 0.0,
        "greedy_avg_4gram_rep": round(np.mean([r["4gram_rep"] for r in greedy]), 4) if greedy else 0.0,
        "greedy_eos_rate": round(np.mean([r["hit_eos"] for r in greedy]), 4) if greedy else 0.0,
        "greedy_avg_tokens": round(np.mean([r["generated_tokens"] for r in greedy]), 1) if greedy else 0.0,
        "sampled_avg_3gram_rep": round(np.mean([r["3gram_rep"] for r in sampled]), 4) if sampled else 0.0,
        "sampled_eos_rate": round(np.mean([r["hit_eos"] for r in sampled]), 4) if sampled else 0.0,
        "sampled_avg_tokens": round(np.mean([r["generated_tokens"] for r in sampled]), 1) if sampled else 0.0,
        "greedy_avg_distinct_1": round(float(np.mean([r["distinct_1"] for r in greedy])), 4) if greedy else 0.0,
        "greedy_avg_distinct_2": round(float(np.mean([r["distinct_2"] for r in greedy])), 4) if greedy else 0.0,
        "greedy_avg_distinct_3": round(float(np.mean([r["distinct_3"] for r in greedy])), 4) if greedy else 0.0,
        "sampled_avg_distinct_2": round(float(np.mean([r["distinct_2"] for r in sampled])), 4) if sampled else 0.0,
        "token_count_min": int(np.min([r["generated_tokens"] for r in results])) if results else 0,
        "token_count_max": int(np.max([r["generated_tokens"] for r in results])) if results else 0,
        "token_count_p25": int(np.percentile([r["generated_tokens"] for r in results], 25)) if results else 0,
        "token_count_p75": int(np.percentile([r["generated_tokens"] for r in results], 75)) if results else 0,
        "elapsed_sec": round(elapsed, 1),
    }

    print(
        f"[GEN {device}] DONE greedy 3gram_rep={summary['greedy_avg_3gram_rep']:.4f}, "
        f"eos_rate={summary['greedy_eos_rate']:.2%}, {elapsed:.1f}s"
    )
    return {"summary": summary, "samples": results}


def eval_repetition_grid(device: str) -> dict:
    """Grid search over 12 generation parameter combinations x 5 prompts.

    Evaluates each config (temperature x repetition_penalty) on the first 5
    prompts and returns results sorted by average 3-gram repetition rate.

    Args:
        device: CUDA device string, e.g. "cuda:5".

    Returns:
        Dict with keys:
          - grid_results: list of per-config dicts, sorted by avg_3gram_rep
          - best: config with lowest avg_3gram_rep
          - elapsed_sec: wall-clock time
    """
    torch.cuda.set_device(int(device.split(":")[-1]))
    print(f"[REP {device}] Loading model...")
    model = _load_model(device)
    tokenizer = _load_tokenizer()
    t0 = time.time()

    rep_prompts = PROMPTS[:5]  # first 5 prompts
    results: list[dict] = []

    total = len(REP_GRID) * len(rep_prompts)
    done = 0

    if USE_CHAT_TEMPLATE:
        print(f"[REP {device}] Chat template ENABLED", flush=True)

    for params in REP_GRID:
        combo_results: list[dict] = []
        for prompt in rep_prompts:
            effective_prompt = CHAT_TEMPLATE_FMT.format(prompt=prompt) if USE_CHAT_TEMPLATE else prompt
            with torch.inference_mode():
                text, n_tokens, hit_eos = generate_one(
                    model,
                    tokenizer,
                    effective_prompt,
                    temperature=params["temperature"],
                    repetition_penalty=params["repetition_penalty"],
                    device=device,
                    max_new_tokens=256,
                )
            combo_results.append(
                {
                    "prompt": prompt,
                    "n_tokens": n_tokens,
                    "hit_eos": hit_eos,
                    "1gram_rep": compute_ngram_rep(text, 1),
                    "2gram_rep": compute_ngram_rep(text, 2),
                    "3gram_rep": compute_ngram_rep(text, 3),
                    "4gram_rep": compute_ngram_rep(text, 4),
                }
            )
            done += 1

        if not combo_results:
            logger.warning("All prompts failed for config %s — skipping", params.get("name", "unknown"))
            continue

        avg_3gram = float(np.mean([r["3gram_rep"] for r in combo_results]))
        avg_4gram = float(np.mean([r["4gram_rep"] for r in combo_results]))
        eos_rate = float(np.mean([r["hit_eos"] for r in combo_results]))
        avg_tokens = float(np.mean([r["n_tokens"] for r in combo_results]))

        entry = {
            "params": params["name"],
            "temperature": params["temperature"],
            "repetition_penalty": params["repetition_penalty"],
            "avg_3gram_rep": round(avg_3gram, 4),
            "avg_4gram_rep": round(avg_4gram, 4),
            "eos_rate": round(eos_rate, 4),
            "avg_tokens": round(avg_tokens, 1),
            "per_prompt": combo_results,
        }
        results.append(entry)
        print(
            f"[REP {device}] {params['name']}: "
            f"3gram={avg_3gram:.2%}, 4gram={avg_4gram:.2%}, "
            f"eos={eos_rate:.0%}, {avg_tokens:.0f}tok"
        )

    elapsed = time.time() - t0

    # Sort by avg 3-gram repetition (ascending = better)
    sorted_results = sorted(results, key=lambda r: r["avg_3gram_rep"])
    best = sorted_results[0]

    print(
        f"[REP {device}] DONE best={best['params']} "
        f"(3gram={best['avg_3gram_rep']:.2%}), {elapsed:.1f}s"
    )
    return {
        "grid_results": sorted_results,
        "best": {
            "params": best["params"],
            "temperature": best["temperature"],
            "repetition_penalty": best["repetition_penalty"],
            "avg_3gram_rep": best["avg_3gram_rep"],
            "avg_4gram_rep": best["avg_4gram_rep"],
        },
        "elapsed_sec": round(elapsed, 1),
    }
