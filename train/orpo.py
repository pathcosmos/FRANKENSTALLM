"""
ORPO (Odds Ratio Preference Optimization) training script.
Uses TRL 0.29.0 ORPOTrainer/ORPOConfig (trl.experimental.orpo).
Optimized for 8x NVIDIA B200 GPUs (183GB VRAM each, ~1.47TB total).

Usage:
    # Full training (8 GPU DDP)
    torchrun --nproc_per_node=8 train/orpo.py \
        --config configs/korean_3b_orpo.yaml

    # Quick test (200 steps)
    python train/orpo.py --config configs/korean_3b_orpo.yaml --max_steps 200

    # Single GPU test
    python train/orpo.py --config configs/korean_3b_orpo.yaml --device cuda:0

Prerequisites:
    pip install trl==0.29.0 transformers accelerate peft datasets
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
import signal as _signal_mod
import sys
import time
import traceback
from pathlib import Path

import torch
from datasets import Dataset, load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    EarlyStoppingCallback,
    TrainerCallback,
)

# TRL imports -- ORPOTrainer/ORPOConfig (TRL 0.29.0, experimental path)
try:
    from trl.experimental.orpo import ORPOConfig, ORPOTrainer
except ImportError:
    print("ERROR: trl not installed or outdated. Run: pip install trl==0.29.0")
    sys.exit(1)

# Telegram notifications
try:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from scripts.telegram_notify import send_telegram_safe
    HAS_TELEGRAM = True
except ImportError:
    HAS_TELEGRAM = False
    def send_telegram_safe(msg, **kw): return False

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("orpo")


# ---------------------------------------------------------------------------
# Custom callback for detailed monitoring
# ---------------------------------------------------------------------------
class ORPOMonitorCallback(TrainerCallback):
    """Monitors ORPO-specific metrics and sends alerts on anomalies."""

    def __init__(self, alert_fn=send_telegram_safe):
        self.alert_fn = alert_fn
        self.start_time = None
        self.last_eval_loss = None
        self.eval_loss_increases = 0
        self.negative_margin_streak = 0

    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = time.time()
        log.info("ORPO training begin -- monitoring active")

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        step = state.global_step

        # Monitor rewards/margins
        margin = logs.get("rewards/margins")
        if margin is not None:
            if margin < 0:
                self.negative_margin_streak += 1
                if self.negative_margin_streak >= 10:
                    msg = (f"[ORPO ALERT] rewards/margins negative for "
                           f"{self.negative_margin_streak} consecutive logs at step {step} "
                           f"(margin={margin:.4f})")
                    log.warning(msg)
                    self.alert_fn(msg)
            else:
                self.negative_margin_streak = 0

        # Log key metrics every logging step
        loss = logs.get("loss")
        chosen = logs.get("rewards/chosen")
        rejected = logs.get("rewards/rejected")
        if loss is not None:
            elapsed = time.time() - self.start_time if self.start_time else 0
            log.info(
                f"step={step} loss={loss:.4f} "
                f"margin={margin if margin is not None else 'N/A'} "
                f"chosen={chosen if chosen is not None else 'N/A'} "
                f"rejected={rejected if rejected is not None else 'N/A'} "
                f"elapsed={elapsed/3600:.1f}h"
            )

        # Check for NaN/Inf
        if loss is not None and (not isinstance(loss, (int, float)) or loss != loss):
            msg = f"[ORPO CRITICAL] NaN/Inf loss detected at step {step}!"
            log.error(msg)
            self.alert_fn(msg)

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is None:
            return
        eval_loss = metrics.get("eval_loss")
        step = state.global_step

        if eval_loss is not None:
            log.info(f"[EVAL] step={step} eval_loss={eval_loss:.4f}")
            if self.last_eval_loss is not None and eval_loss > self.last_eval_loss:
                self.eval_loss_increases += 1
                log.warning(
                    f"[EVAL] eval_loss increased: {self.last_eval_loss:.4f} -> {eval_loss:.4f} "
                    f"({self.eval_loss_increases}/3 before early stop)"
                )
            else:
                self.eval_loss_increases = 0
            self.last_eval_loss = eval_loss

    def on_train_end(self, args, state, control, **kwargs):
        elapsed = time.time() - self.start_time if self.start_time else 0
        log.info(f"ORPO training ended -- total time: {elapsed/3600:.2f}h, "
                 f"total steps: {state.global_step}")

    def on_save(self, args, state, control, **kwargs):
        log.info(f"Checkpoint saved at step {state.global_step}")


def load_hf_preference_dataset(dataset_name: str, token: str | None = None) -> Dataset:
    """Load and normalize a HuggingFace preference dataset to {prompt, chosen, rejected}."""
    ds = load_dataset(dataset_name, split="train", token=token)

    # kuotient/orca-math-korean-dpo-pairs format: {system, question, chosen, rejected}
    if "question" in ds.column_names and "chosen" in ds.column_names:
        def normalize(example):
            prompt = example.get("system", "") + "\n" + example["question"]
            return {"prompt": prompt.strip(), "chosen": example["chosen"], "rejected": example["rejected"]}
        return ds.map(normalize, remove_columns=ds.column_names)

    # nayohan/preference-collection-ko-full format: {response_A, response_B, orig_preference}
    if "orig_preference" in ds.column_names:
        def normalize_pref(example):
            prompt = example.get("orig_instruction", example.get("instruction", ""))
            if example["orig_preference"] == "B":
                return {"prompt": prompt, "chosen": example["orig_response_B"], "rejected": example["orig_response_A"]}
            else:
                return {"prompt": prompt, "chosen": example["orig_response_A"], "rejected": example["orig_response_B"]}
        return ds.map(normalize_pref, remove_columns=ds.column_names)

    # Already in {prompt, chosen, rejected} format
    if all(c in ds.column_names for c in ["prompt", "chosen", "rejected"]):
        return ds

    raise ValueError(f"Unknown dataset format. Columns: {ds.column_names}")


def load_custom_jsonl(path: str) -> Dataset:
    """Load custom JSONL with {prompt, chosen, rejected} fields."""
    data = []
    with open(path) as f:
        for line in f:
            data.append(json.loads(line))
    return Dataset.from_list(data)


def load_yaml_config(path: str) -> dict:
    """Load YAML config and return as dict."""
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="ORPO Training (TRL 0.29.0 -- 8xB200 optimized)")
    parser.add_argument("--config", type=str, default=None, help="YAML config file path")
    parser.add_argument("--model_path", type=str, default=None, help="HF format model path")
    parser.add_argument("--dataset", type=str, default="kuotient/orca-math-korean-dpo-pairs")
    parser.add_argument("--custom_data_path", type=str, default=None, help="Custom JSONL preference data")
    parser.add_argument("--output_dir", type=str, default="checkpoints/korean_3b_orpo")
    parser.add_argument("--hf_token", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--beta", type=float, default=0.1, help="ORPO beta (odds ratio weight)")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--max_length", type=int, default=1536)
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--eval_split_ratio", type=float, default=0.05, help="Fraction of data for eval")
    parser.add_argument("--eval_steps", type=int, default=500)
    parser.add_argument("--early_stopping_patience", type=int, default=3)
    parser.add_argument("--max_steps", type=int, default=-1, help="Override max steps (for quick test)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_total_limit", type=int, default=5)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True)
    parser.add_argument("--report_to", type=str, default="none")
    parser.add_argument("--dataset_num_proc", type=int, default=8,
                        help="Number of processes for parallel tokenization in ORPOTrainer")
    parser.add_argument("--dataloader_num_workers", type=int, default=4,
                        help="Number of dataloader worker processes")
    parser.add_argument("--no_load_best", action="store_true", default=False,
                        help="Disable load_best_model_at_end (for sweep/quick tests)")
    args = parser.parse_args()

    # Override CLI defaults with YAML config values
    if args.config:
        cfg = load_yaml_config(args.config)
        for key, value in cfg.items():
            if hasattr(args, key):
                setattr(args, key, value)

    if not args.model_path:
        parser.error("--model_path is required (or set model_path in YAML config)")

    # Log all resolved config
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    is_main = local_rank == 0
    if is_main:
        log.info("=" * 70)
        log.info("ORPO Training Configuration (8xB200 optimized)")
        log.info("=" * 70)
        for k, v in sorted(vars(args).items()):
            log.info(f"  {k}: {v}")
        log.info("=" * 70)

        # GPU info
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                mem = torch.cuda.get_device_properties(i).total_memory / 1e9
                log.info(f"  GPU {i}: {torch.cuda.get_device_name(i)} ({mem:.1f} GB)")

    # Validate paths
    if not Path(args.model_path).exists():
        raise FileNotFoundError(f"Model path not found: {args.model_path}")
    if args.custom_data_path and not Path(args.custom_data_path).exists():
        raise FileNotFoundError(f"Data path not found: {args.custom_data_path}")

    # NCCL/DDP environment diagnostics
    if is_main:
        log.info("--- DDP/NCCL Environment ---")
        for env_key in ["RANK", "WORLD_SIZE", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT",
                        "NCCL_IB_DISABLE", "NCCL_BUFFSIZE", "NCCL_P2P_LEVEL",
                        "OMP_NUM_THREADS", "PYTORCH_CUDA_ALLOC_CONF"]:
            log.info(f"  {env_key}={os.environ.get(env_key, '(not set)')}")
        log.info(f"  torch.distributed.is_available={torch.distributed.is_available()}")
        if torch.distributed.is_initialized():
            log.info(f"  world_size={torch.distributed.get_world_size()}, "
                     f"rank={torch.distributed.get_rank()}")

    # Load model (bfloat16 + flash_attention_2 for B200)
    log.info(f"Loading model from {args.model_path}...")
    t0 = time.time()
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )
    except Exception as e:
        log.error(f"Model loading failed: {e}")
        send_telegram_safe(f"[ORPO FATAL] Model load failed: {e}")
        raise
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if is_main:
        n_params = sum(p.numel() for p in model.parameters())
        log.info(f"Model loaded: {n_params:,} params in {time.time()-t0:.1f}s")
        log.info(f"Tokenizer: vocab_size={tokenizer.vocab_size}, "
                 f"pad_token='{tokenizer.pad_token}', eos_token='{tokenizer.eos_token}'")

    # Load dataset
    t0 = time.time()
    try:
        if args.custom_data_path:
            log.info(f"Loading custom data from {args.custom_data_path}...")
            fsize_mb = Path(args.custom_data_path).stat().st_size / 1e6
            log.info(f"  File size: {fsize_mb:.1f} MB")
            dataset = load_custom_jsonl(args.custom_data_path)
        else:
            log.info(f"Loading dataset {args.dataset}...")
            dataset = load_hf_preference_dataset(args.dataset, token=args.hf_token)
    except Exception as e:
        log.error(f"Dataset loading failed: {e}")
        send_telegram_safe(f"[ORPO FATAL] Data load failed: {e}")
        raise

    if is_main:
        log.info(f"Dataset loaded: {len(dataset)} pairs in {time.time()-t0:.1f}s")
        # Data quality check
        sample = dataset[0]
        log.info(f"Sample keys: {list(sample.keys())}")
        for key in ["prompt", "chosen", "rejected"]:
            if key not in sample:
                raise ValueError(f"Dataset missing required column: {key}")
            val = sample[key]
            log.info(f"  {key}: {str(val)[:100]}...")

        # Length distribution check (sample first 1000)
        sample_size = min(1000, len(dataset))
        lengths = [len(str(dataset[i]["prompt"])) + max(len(str(dataset[i]["chosen"])),
                   len(str(dataset[i]["rejected"]))) for i in range(sample_size)]
        avg_len = sum(lengths) / len(lengths)
        max_len = max(lengths)
        log.info(f"  Char lengths (sample {sample_size}): avg={avg_len:.0f}, max={max_len}")

    # Filter out samples where prompt is too long for the response to fit in max_length.
    # Without this, samples with 0 response tokens cause NaN in ORPO log-probability computation
    # (division by zero in average_log_prob when loss_mask is all-zero).
    # Also catches TRL truncation bug: tokenize_row uses longer_response_length = max(chosen_len, rejected_len)
    # and truncates BOTH responses to [:max_length - longer_response_length]. When longer >= max_length,
    # the shorter response becomes EMPTY → NaN.
    pre_filter = len(dataset)
    def _has_response_room(example):
        prompt_tok_len = len(tokenizer.encode(example["prompt"], add_special_tokens=False))
        chosen_tok_len = len(tokenizer.encode(example["chosen"], add_special_tokens=False))
        rejected_tok_len = len(tokenizer.encode(example["rejected"], add_special_tokens=False))

        # 1. Prompt must leave room for at least 16 response tokens
        if prompt_tok_len + 16 > args.max_length:
            return False

        # 2. Each response independently must fit with prompt
        # (TRL adds BOS/EOS, so use +2 margin)
        if prompt_tok_len + chosen_tok_len + 2 > args.max_length * 2:
            return False  # extremely long, will cause issues
        if prompt_tok_len + rejected_tok_len + 2 > args.max_length * 2:
            return False

        # 3. The longer response must not exceed max_length alone
        # (TRL bug: both responses truncated by max(chosen_len, rejected_len))
        longer = max(chosen_tok_len, rejected_tok_len)
        if longer >= args.max_length:
            return False

        return True
    dataset = dataset.filter(_has_response_room, num_proc=min(args.dataset_num_proc, 32) if is_main else 1)
    if is_main:
        log.info(f"Filtered: {pre_filter:,} -> {len(dataset):,} "
                 f"(removed {pre_filter - len(dataset):,} samples with prompt > max_length-16 or TRL truncation risk)")

    # Train/eval split
    split = dataset.train_test_split(test_size=args.eval_split_ratio, seed=args.seed)
    train_dataset = split["train"]
    eval_dataset = split["test"]
    log.info(f"Train: {len(train_dataset):,}, Eval: {len(eval_dataset):,}")

    # Compute training stats for warmup_steps calculation
    n_gpus = max(torch.cuda.device_count(), 1) if torch.cuda.is_available() else 1
    eff_batch = args.batch_size * args.gradient_accumulation_steps * n_gpus
    steps_per_epoch = len(train_dataset) // eff_batch
    total_steps = args.max_steps if args.max_steps > 0 else steps_per_epoch * args.epochs
    computed_warmup_steps = int(total_steps * args.warmup_ratio)
    if is_main:
        log.info(f"Training plan: eff_batch={eff_batch}, steps/epoch={steps_per_epoch:,}, "
                 f"total={total_steps:,}, warmup={computed_warmup_steps}")

    # DDP tokenization strategy:
    # TRL ORPOTrainer uses main_process_first() — rank 0 tokenizes first, then ranks 1-7.
    # With multiprocessing (num_proc>1), ranks 1-7 all spawn workers simultaneously,
    # causing CPU/memory oversubscription (e.g. 7 ranks × 8 workers = 56 processes).
    # Fix: rank 0 uses full num_proc for speed, other ranks use 1 (should hit cache).
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if world_size > 1:
        effective_num_proc = args.dataset_num_proc if is_main else 1
        if is_main:
            log.info(f"DDP tokenization: rank 0 uses num_proc={args.dataset_num_proc}, "
                     f"other {world_size-1} ranks use num_proc=1 (cache)")
    else:
        effective_num_proc = args.dataset_num_proc

    # ORPOConfig (TRL 0.29.0) -- optimized for 8x B200
    orpo_config = ORPOConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.lr,
        beta=args.beta,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_steps=computed_warmup_steps,
        weight_decay=args.weight_decay,
        bf16=args.bf16,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        max_length=args.max_length,
        gradient_checkpointing=args.gradient_checkpointing,
        report_to=args.report_to,
        remove_unused_columns=False,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        metric_for_best_model="eval_loss" if not args.no_load_best else None,
        load_best_model_at_end=not args.no_load_best,
        greater_is_better=False if not args.no_load_best else None,
        max_steps=args.max_steps,
        seed=args.seed,
        # B200 hardware optimizations
        dataloader_num_workers=args.dataloader_num_workers,
        dataloader_pin_memory=True,
        ddp_find_unused_parameters=False,
        ddp_timeout=7200,  # 2h — tokenization takes ~30min on 683K samples
        dataset_num_proc=effective_num_proc,
    )

    # ORPOTrainer (no reference model needed)
    log.info("Initializing ORPOTrainer (tokenization will happen here — may take a while)...")
    t0 = time.time()
    monitor = ORPOMonitorCallback()
    try:
        trainer = ORPOTrainer(
            model=model,
            args=orpo_config,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=tokenizer,
            callbacks=[cb for cb in [
                EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)
                    if not args.no_load_best else None,
                monitor,
            ] if cb is not None],
        )
    except Exception as e:
        log.error(f"ORPOTrainer init failed: {e}\n{traceback.format_exc()}")
        send_telegram_safe(f"[ORPO FATAL] Trainer init failed: {e}")
        raise
    if is_main:
        log.info(f"ORPOTrainer initialized in {time.time()-t0:.1f}s "
                 f"(dataset_num_proc={args.dataset_num_proc})")

    # SIGHUP/SIGTERM defense -- graceful shutdown with emergency checkpoint
    def _graceful_shutdown_handler(signum, frame):
        sig_name = _signal_mod.Signals(signum).name
        log.warning(f"Received {sig_name}. Saving emergency checkpoint...")
        try:
            emergency_path = os.path.join(args.output_dir, "emergency_checkpoint")
            trainer.save_model(emergency_path)
            log.info(f"Emergency checkpoint saved to {emergency_path}")
            send_telegram_safe(
                f"[ORPO] Signal {sig_name} received at step {trainer.state.global_step}. "
                f"Emergency checkpoint saved."
            )
        except Exception as e:
            log.error(f"Emergency save failed: {e}")
            send_telegram_safe(f"[ORPO] Emergency save FAILED after {sig_name}: {e}")
        sys.exit(1)

    for _sig in (_signal_mod.SIGHUP, _signal_mod.SIGTERM):
        _signal_mod.signal(_sig, _graceful_shutdown_handler)

    # Pre-training VRAM report
    if is_main and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        alloc = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        log.info(f"Pre-train VRAM: allocated={alloc:.1f}GB, reserved={reserved:.1f}GB")

    start_msg = (
        f"[ORPO] Training started\n"
        f"  model: {args.model_path}\n"
        f"  beta: {args.beta}, lr: {args.lr}\n"
        f"  train: {len(train_dataset):,}, eval: {len(eval_dataset):,}\n"
        f"  eff_batch: {eff_batch}, steps/epoch: {steps_per_epoch:,}, total: {total_steps:,}\n"
        f"  warmup: {computed_warmup_steps} steps ({args.warmup_ratio*100:.0f}%)\n"
        f"  max_length: {args.max_length}, max_steps: {args.max_steps}\n"
        f"  dataset_num_proc: {args.dataset_num_proc}, dl_workers: {args.dataloader_num_workers}"
    )
    log.info(start_msg.replace("[ORPO] ", ""))
    send_telegram_safe(start_msg)

    try:
        trainer.train()

        # Post-training VRAM report
        if is_main and torch.cuda.is_available():
            peak = torch.cuda.max_memory_allocated() / 1e9
            log.info(f"Peak VRAM usage: {peak:.1f}GB")

        trainer.save_model(args.output_dir)
        log.info(f"Model saved to {args.output_dir}")

        # Extract final metrics
        final_metrics = {}
        for entry in reversed(trainer.state.log_history):
            if "loss" in entry and "loss" not in final_metrics:
                final_metrics["loss"] = entry["loss"]
            if "eval_loss" in entry and "eval_loss" not in final_metrics:
                final_metrics["eval_loss"] = entry["eval_loss"]
            if "rewards/margins" in entry and "rewards/margins" not in final_metrics:
                final_metrics["rewards/margins"] = entry["rewards/margins"]
            if len(final_metrics) >= 3:
                break

        done_msg = (
            f"[ORPO] Training complete!\n"
            f"  output: {args.output_dir}\n"
            f"  steps: {trainer.state.global_step}\n"
            f"  final loss: {final_metrics.get('loss', 'N/A')}\n"
            f"  final eval_loss: {final_metrics.get('eval_loss', 'N/A')}\n"
            f"  final margins: {final_metrics.get('rewards/margins', 'N/A')}"
        )
        log.info(done_msg.replace("[ORPO] ", ""))
        send_telegram_safe(done_msg)

    except KeyboardInterrupt:
        log.warning("Training interrupted by user (KeyboardInterrupt)")
        send_telegram_safe(f"[ORPO] Training interrupted at step {trainer.state.global_step}")
        trainer.save_model(os.path.join(args.output_dir, "interrupted_checkpoint"))
        log.info("Interrupted checkpoint saved.")

    except Exception as e:
        tb = traceback.format_exc()
        error_msg = f"[ORPO] Training FAILED at step {trainer.state.global_step}: {e}"
        log.error(f"{error_msg}\n{tb}")
        send_telegram_safe(f"{error_msg}\n{tb[:500]}")
        # Try emergency save
        try:
            trainer.save_model(os.path.join(args.output_dir, "error_checkpoint"))
            log.info("Error checkpoint saved.")
        except Exception:
            log.error("Error checkpoint save also failed.")
        raise


if __name__ == "__main__":
    main()
