#!/usr/bin/env python3
"""Fix GGUF newline crash by adding byte-fallback tokens to the tokenizer.

Problem: The SentencePiece Unigram tokenizer was trained without byte_fallback=True,
so characters like \n have no token representation. llama.cpp crashes when it
encounters these characters because there's no byte-fallback.

Fix:
  1. Add 256 byte-fallback tokens (<0x00> .. <0xFF>) to tokenizer.json
  2. Resize model embeddings from 64000 -> 64256
  3. Update config.json vocab_size
  4. Copy tokenizer.model for proper GGUF conversion

Usage:
    python scripts/fix_tokenizer_byte_fallback.py \
        --input outputs/hf_checkpoint-best \
        --output outputs/hf_checkpoint-best-fixed \
        --sp_model tokenizer/korean_sp/tokenizer.model
"""

import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


BYTE_FALLBACK_COUNT = 256
BYTE_TOKEN_TEMPLATE = "<0x{:02X}>"


def fix_tokenizer_json(input_path: Path, output_path: Path):
    """Add byte_fallback=True and 256 byte tokens to tokenizer.json."""
    with open(input_path) as f:
        tok = json.load(f)

    model = tok["model"]
    vocab = model["vocab"]  # list of [piece, score]
    original_size = len(vocab)

    # Enable byte_fallback
    model["byte_fallback"] = True

    # Add 256 byte tokens with very low score (they're fallback only)
    for i in range(BYTE_FALLBACK_COUNT):
        byte_token = BYTE_TOKEN_TEMPLATE.format(i)
        vocab.append([byte_token, 0.0])

    new_size = len(vocab)
    print(f"  Vocab: {original_size} -> {new_size} (+{BYTE_FALLBACK_COUNT} byte tokens)")
    print(f"  byte_fallback: False -> True")

    # Also add byte tokens to added_tokens list
    added = tok.get("added_tokens", [])
    for i in range(BYTE_FALLBACK_COUNT):
        byte_token = BYTE_TOKEN_TEMPLATE.format(i)
        added.append({
            "id": original_size + i,
            "content": byte_token,
            "single_word": False,
            "lstrip": False,
            "rstrip": False,
            "normalized": False,
            "special": True,
        })
    tok["added_tokens"] = added

    with open(output_path, "w") as f:
        json.dump(tok, f, ensure_ascii=False, indent=2)

    return original_size, new_size


def fix_config_json(input_path: Path, output_path: Path, new_vocab_size: int):
    """Update vocab_size in config.json."""
    with open(input_path) as f:
        config = json.load(f)

    old_size = config["vocab_size"]
    config["vocab_size"] = new_vocab_size
    print(f"  config.json vocab_size: {old_size} -> {new_vocab_size}")

    with open(output_path, "w") as f:
        json.dump(config, f, indent=2)


def resize_embeddings(input_path: Path, output_path: Path,
                      old_vocab: int, new_vocab: int, tie_embeddings: bool):
    """Resize embedding and lm_head weights to accommodate new tokens."""
    print(f"  Loading model weights from {input_path} ...")
    state_dict = load_file(str(input_path))

    embed_key = "model.embed_tokens.weight"
    lm_head_key = "lm_head.weight"

    if embed_key not in state_dict:
        raise KeyError(f"{embed_key} not found in state_dict. Keys: {list(state_dict.keys())[:10]}")

    embed = state_dict[embed_key]
    print(f"  embed_tokens shape: {embed.shape}")

    hidden_size = embed.shape[1]
    extra = new_vocab - old_vocab

    # Initialize new embeddings as mean of existing (better than random for byte tokens)
    mean_embed = embed.mean(dim=0, keepdim=True)
    # Add small noise to avoid identical embeddings
    noise = torch.randn(extra, hidden_size, dtype=embed.dtype) * 0.01
    new_rows = mean_embed.expand(extra, -1) + noise

    new_embed = torch.cat([embed, new_rows], dim=0)
    state_dict[embed_key] = new_embed
    print(f"  embed_tokens resized: {embed.shape} -> {new_embed.shape}")

    if tie_embeddings:
        # When tie_word_embeddings=True, lm_head shares embed_tokens
        # Remove lm_head if present (it will be tied automatically)
        if lm_head_key in state_dict:
            del state_dict[lm_head_key]
            print(f"  lm_head removed (tie_word_embeddings=True)")
    else:
        if lm_head_key in state_dict:
            lm_head = state_dict[lm_head_key]
            mean_lm = lm_head.mean(dim=0, keepdim=True)
            noise_lm = torch.randn(extra, hidden_size, dtype=lm_head.dtype) * 0.01
            new_lm = torch.cat([lm_head, mean_lm.expand(extra, -1) + noise_lm], dim=0)
            state_dict[lm_head_key] = new_lm
            print(f"  lm_head resized: {lm_head.shape} -> {new_lm.shape}")

    print(f"  Saving to {output_path} ...")
    save_file(state_dict, str(output_path))


def main():
    parser = argparse.ArgumentParser(description="Fix tokenizer byte-fallback for GGUF")
    parser.add_argument("--input", type=Path, required=True, help="Input HF checkpoint dir")
    parser.add_argument("--output", type=Path, required=True, help="Output fixed HF checkpoint dir")
    parser.add_argument("--sp_model", type=Path, default=None,
                        help="SentencePiece .model file to copy (for GGUF conversion)")
    args = parser.parse_args()

    input_dir = args.input
    output_dir = args.output

    if not input_dir.exists():
        print(f"ERROR: Input directory not found: {input_dir}")
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load config to check tie_word_embeddings
    with open(input_dir / "config.json") as f:
        config = json.load(f)
    old_vocab = config["vocab_size"]
    new_vocab = old_vocab + BYTE_FALLBACK_COUNT
    tie_embeddings = config.get("tie_word_embeddings", False)

    print(f"=== Byte-Fallback Fix ===")
    print(f"Input:  {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Old vocab: {old_vocab}, New vocab: {new_vocab}")
    print(f"tie_word_embeddings: {tie_embeddings}")
    print()

    # 1. Fix tokenizer.json
    print("[1/4] Fixing tokenizer.json ...")
    fix_tokenizer_json(
        input_dir / "tokenizer.json",
        output_dir / "tokenizer.json",
    )

    # 2. Fix config.json
    print("[2/4] Fixing config.json ...")
    fix_config_json(
        input_dir / "config.json",
        output_dir / "config.json",
        new_vocab,
    )

    # 3. Resize model weights
    print("[3/4] Resizing embeddings ...")
    resize_embeddings(
        input_dir / "model.safetensors",
        output_dir / "model.safetensors",
        old_vocab, new_vocab, tie_embeddings,
    )

    # 4. Copy other files
    print("[4/4] Copying remaining files ...")
    for fname in ["tokenizer_config.json", "generation_config.json"]:
        src = input_dir / fname
        if src.exists():
            shutil.copy2(src, output_dir / fname)
            print(f"  Copied {fname}")

    # Copy SentencePiece model if provided (needed for GGUF conversion)
    if args.sp_model and args.sp_model.exists():
        shutil.copy2(args.sp_model, output_dir / "tokenizer.model")
        print(f"  Copied tokenizer.model from {args.sp_model}")
    elif (input_dir / "tokenizer.model").exists():
        shutil.copy2(input_dir / "tokenizer.model", output_dir / "tokenizer.model")
        print(f"  Copied tokenizer.model from input dir")

    # Update tokenizer_config.json to add added_tokens_decoder for byte tokens
    tc_path = output_dir / "tokenizer_config.json"
    if tc_path.exists():
        with open(tc_path) as f:
            tc = json.load(f)
        added_tokens_decoder = tc.get("added_tokens_decoder", {})
        for i in range(BYTE_FALLBACK_COUNT):
            token_id = old_vocab + i
            byte_token = BYTE_TOKEN_TEMPLATE.format(i)
            added_tokens_decoder[str(token_id)] = {
                "content": byte_token,
                "lstrip": False,
                "normalized": False,
                "rstrip": False,
                "single_word": False,
                "special": True,
            }
        tc["added_tokens_decoder"] = added_tokens_decoder
        with open(tc_path, "w") as f:
            json.dump(tc, f, indent=2)
        print(f"  Updated tokenizer_config.json with {BYTE_FALLBACK_COUNT} byte tokens")

    print()
    print(f"=== Done! Fixed checkpoint at: {output_dir} ===")
    print(f"Next: python outputs/llama.cpp/convert_hf_to_gguf.py {output_dir} --outfile outputs/gguf/frankenstallm-3b-f16.gguf --outtype f16")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
