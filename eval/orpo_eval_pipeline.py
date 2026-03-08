"""
FRANKENSTALLM 3B — ORPO Evaluation Pipeline Orchestrator
=========================================================

Evaluates the ORPO checkpoint across 6 dimensions and generates a
3-way comparison report (Base vs SFT vs ORPO).

Runs 3 phases sequentially (no Phase 0 — ORPO checkpoints are already HF format):
  Phase 1  — Internal evaluation across 8 GPUs (PPL, Calibration, Generation)
  Phase 2  — Standard benchmarks via lm-eval-harness (8 GPU parallel)
  Phase 3  — Base vs SFT vs ORPO 3-way comparison report generation

Usage:
    python eval/orpo_eval_pipeline.py
    python eval/orpo_eval_pipeline.py --dry-run
    python eval/orpo_eval_pipeline.py --skip-phase1
    python eval/orpo_eval_pipeline.py --checkpoint checkpoints/korean_3b_orpo_v1/checkpoint-1000/
"""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
import os
import re
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Project root
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ---------------------------------------------------------------------------
# ORPO checkpoint and comparison results paths
# ---------------------------------------------------------------------------
ORPO_CHECKPOINT_DIR = _PROJECT_ROOT / "checkpoints" / "korean_3b_orpo_v1"
BASE_RESULTS_DIR = _PROJECT_ROOT / "eval" / "outputs" / "3b_reeval_20260305_1451"
SFT_RESULTS_DIR = _PROJECT_ROOT / "eval" / "outputs" / "3b_sft_eval_20260306_1536"

# Fallback tokenizer
_FALLBACK_TOKENIZER = str(
    _PROJECT_ROOT / "tokenizer" / "korean_sp" / "tokenizer.json"
)

# ---------------------------------------------------------------------------
# Import shared infrastructure from full_eval_pipeline
# ---------------------------------------------------------------------------
from eval.full_eval_pipeline import (
    _bar,
    _build_phase1_tasks,
    _build_phase2_tasks,
    _fmt_seconds,
    _make_output_dir,
    _NUMA_CORES,
    _print_banner,
    _print_phase_header,
    _save_json,
    _spawn_task,
    _wait_and_collect,
    SEQ_LEN,
    STRIDE,
    BATCH_SIZE,
    DATA_DIR,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("orpo_eval")


# ===========================================================================
# ORPO checkpoint auto-detection
# ===========================================================================

def detect_latest_checkpoint(checkpoint_dir: Path) -> Optional[Path]:
    """Find the latest checkpoint-* subdirectory by numeric step."""
    if not checkpoint_dir.exists():
        return None

    candidates = []
    for d in checkpoint_dir.iterdir():
        if d.is_dir() and d.name.startswith("checkpoint-"):
            try:
                step = int(d.name.split("-", 1)[1])
                candidates.append((step, d))
            except ValueError:
                continue

    if not candidates:
        return None

    candidates.sort(key=lambda x: x[0])
    return candidates[-1][1]


def resolve_tokenizer(checkpoint_path: Path) -> str:
    """Find tokenizer: first in checkpoint dir, then fallback."""
    ckpt_tokenizer = checkpoint_path / "tokenizer.json"
    if ckpt_tokenizer.exists():
        return str(ckpt_tokenizer)
    if Path(_FALLBACK_TOKENIZER).exists():
        return _FALLBACK_TOKENIZER
    raise FileNotFoundError(
        f"Tokenizer not found in {checkpoint_path} or {_FALLBACK_TOKENIZER}"
    )


# ===========================================================================
# Training curve extraction
# ===========================================================================

def extract_training_curve(
    train_log_path: Path,
    output_dir: Path,
) -> Dict[str, Any]:
    """Parse train.log to extract training and eval metrics per step.

    Returns dict with {"train_steps": [...], "eval_steps": [...]}.
    Saves to output_dir / "training_curve.json".
    """
    curve: Dict[str, Any] = {"train_steps": [], "eval_steps": []}

    if not train_log_path.exists():
        logger.warning("  train.log not found: %s", train_log_path)
        _save_json(curve, output_dir / "training_curve.json")
        return curve

    logger.info("  Parsing training log: %s", train_log_path)

    # Numeric value pattern — values may be quoted strings: 'loss': '2.339' or bare: 'loss': 2.339
    _NUM = r"'?(?:{})'?"  # template for named group

    # Patterns for training loss lines like: {'loss': '2.339', 'grad_norm': '0.53', ...}
    train_loss_re = re.compile(
        r"\{[^}]*'loss'\s*:\s*'?(?P<loss>[-\d.]+(?:e[+-]?\d+)?)'?"
        r"(?:.*?'grad_norm'\s*:\s*'?(?P<grad_norm>[-\d.]+(?:e[+-]?\d+)?)'?)?"
        r"(?:.*?'learning_rate'\s*:\s*'?(?P<lr>[-\d.]+(?:e[+-]?\d+)?)'?)?"
        r"(?:.*?'rewards/accuracies'\s*:\s*'?(?P<rewards_acc>[-\d.]+(?:e[+-]?\d+)?)'?)?"
        r"(?:.*?'rewards/margins'\s*:\s*'?(?P<rewards_margins>[-\d.]+(?:e[+-]?\d+)?)'?)?"
        r"(?:.*?'nll_loss'\s*:\s*'?(?P<nll_loss>[-\d.]+(?:e[+-]?\d+)?)'?)?"
        r"(?:.*?'epoch'\s*:\s*'?(?P<epoch>[-\d.]+(?:e[+-]?\d+)?)'?)?"
    )

    # Patterns for eval lines like: {'eval_loss': '1.713', 'eval_rewards/chosen': '-0.36', ...}
    eval_loss_re = re.compile(
        r"\{[^}]*'eval_loss'\s*:\s*'?(?P<eval_loss>[-\d.]+(?:e[+-]?\d+)?)'?"
        r"(?:.*?'eval_rewards/chosen'\s*:\s*'?(?P<rewards_chosen>[-\d.]+(?:e[+-]?\d+)?)'?)?"
        r"(?:.*?'eval_rewards/rejected'\s*:\s*'?(?P<rewards_rejected>[-\d.]+(?:e[+-]?\d+)?)'?)?"
        r"(?:.*?'eval_rewards/accuracies'\s*:\s*'?(?P<rewards_accuracies>[-\d.]+(?:e[+-]?\d+)?)'?)?"
        r"(?:.*?'eval_rewards/margins'\s*:\s*'?(?P<rewards_margins>[-\d.]+(?:e[+-]?\d+)?)'?)?"
        r"(?:.*?'eval_nll_loss'\s*:\s*'?(?P<nll_loss>[-\d.]+(?:e[+-]?\d+)?)'?)?"
        r"(?:.*?'eval_log_odds_ratio'\s*:\s*'?(?P<log_odds_ratio>[-\d.]+(?:e[+-]?\d+)?)'?)?"
        r"(?:.*?'eval_runtime'\s*:\s*'?(?P<runtime>[-\d.]+(?:e[+-]?\d+)?)'?)?"
        r"(?:.*?'epoch'\s*:\s*'?(?P<epoch>[-\d.]+(?:e[+-]?\d+)?)'?)?"
    )

    # Step counter pattern — look for step in same line or progress bar like "1000/9840"
    step_re = re.compile(r"'(?:global_)?step'\s*:\s*(\d+)")
    # Progress bar step: " 10%|█         | 1000/9840 [35:34..."
    # These appear as \r-separated segments on the same line
    progress_re = re.compile(r"\|\s*(\d+)/\d+\s+\[")

    train_step_counter = 0
    eval_step_counter = 0

    with open(train_log_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            # Extract the latest progress bar step from this line (may have many \r segments)
            all_prog_steps = progress_re.findall(line)
            if all_prog_steps:
                # Take the last (highest) progress bar step on this line
                train_step_counter = max(int(s) for s in all_prog_steps)

            # Try eval match first (eval lines also contain 'loss' key)
            eval_m = eval_loss_re.search(line)
            if eval_m:
                # For eval entries, infer step from epoch since progress bar shows eval iterator steps
                epoch_val = eval_m.group("epoch")
                if epoch_val:
                    # step ≈ epoch / (1 / total_train_steps) — for ~1 epoch training
                    # Use the last known training step as reference
                    step = round(float(epoch_val) * 9840)  # 9840 total steps
                else:
                    step_m = step_re.search(line)
                    step = int(step_m.group(1)) if step_m else train_step_counter
                eval_step_counter = step

                entry: Dict[str, Any] = {"step": step}
                for key in ("eval_loss", "rewards_chosen", "rewards_rejected",
                            "rewards_accuracies", "rewards_margins",
                            "nll_loss", "log_odds_ratio", "runtime", "epoch"):
                    val = eval_m.group(key) if key in eval_m.groupdict() else None
                    if val is not None:
                        entry[key] = float(val)
                curve["eval_steps"].append(entry)
                continue

            # Training loss match
            train_m = train_loss_re.search(line)
            if train_m:
                step_m = step_re.search(line)
                step = int(step_m.group(1)) if step_m else train_step_counter

                entry = {"step": step, "loss": float(train_m.group("loss"))}
                for key in ("grad_norm", "lr", "rewards_acc", "rewards_margins",
                            "nll_loss", "epoch"):
                    val = train_m.group(key)
                    if val is not None:
                        entry[key] = float(val)
                curve["train_steps"].append(entry)

    logger.info(
        "  Extracted %d train steps, %d eval steps from log.",
        len(curve["train_steps"]),
        len(curve["eval_steps"]),
    )

    out_path = output_dir / "training_curve.json"
    _save_json(curve, out_path)
    logger.info("  Training curve saved: %s", out_path)
    return curve


# ===========================================================================
# Override: spawn tasks with ORPO environment variables
# ===========================================================================

def _spawn_orpo_task(
    task_name: str,
    gpu_id: int,
    output_path: Path,
    label: str,
    checkpoint: str,
    tokenizer: str,
    use_chat_template: bool = False,
    extra_args: Optional[Dict[str, str]] = None,
) -> tuple:
    """Spawn a subprocess task with ORPO checkpoint via environment variables."""
    cmd = [
        sys.executable,
        str(_PROJECT_ROOT / "eval" / "tasks" / "task_runner.py"),
        "--task", task_name,
        "--gpu-id", str(gpu_id),
        "--output", str(output_path),
    ]
    if extra_args:
        for k, v in extra_args.items():
            cmd.extend([k, v])

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["EVAL_CHECKPOINT"] = checkpoint
    env["EVAL_TOKENIZER"] = tokenizer
    if use_chat_template:
        env["USE_CHAT_TEMPLATE"] = "1"

    import subprocess
    output_path.parent.mkdir(parents=True, exist_ok=True)
    log_path = output_path.with_suffix(".log")
    log_file = open(log_path, "w")

    logger.info("  Spawning: %s (GPU %d) [ORPO]", label, gpu_id)
    proc = subprocess.Popen(
        cmd,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        env=env,
        cwd=str(_PROJECT_ROOT),
    )
    return proc, label, output_path, log_file


# ===========================================================================
# Phase 1 — Internal Evaluation (ORPO variant)
# ===========================================================================

def run_orpo_phase1(
    output_dir: Path,
    gpu_ids: List[int],
    checkpoint: str,
    tokenizer: str,
) -> Dict[str, Any]:
    """Run internal eval tasks with ORPO checkpoint, chat template enabled for gen tasks."""
    task_descriptors = _build_phase1_tasks(gpu_ids)
    processes = []

    for desc in task_descriptors:
        is_gen_task = desc["task"] in ("generation", "repetition_grid")
        out_path = output_dir / f"phase1_{desc['task']}_gpu{desc['gpu_id']}.json"
        proc_info = _spawn_orpo_task(
            task_name=desc["task"],
            gpu_id=desc["gpu_id"],
            output_path=out_path,
            label=desc["label"],
            checkpoint=checkpoint,
            tokenizer=tokenizer,
            use_chat_template=is_gen_task,
            extra_args=desc.get("extra_args"),
        )
        processes.append(proc_info)

    results = _wait_and_collect(processes)

    phase1_out = output_dir / "phase1_results.json"
    _save_json(results, phase1_out)
    logger.info("  Phase 1 results saved: %s", phase1_out)

    # Save generation samples separately
    gen_samples: Dict[str, Any] = {}
    for label, result in results.items():
        if isinstance(result, dict) and "error" not in result:
            if "Generation" in label:
                gen_samples["generation"] = result
            elif "Repetition" in label:
                gen_samples["repetition_grid"] = result
    if gen_samples:
        gen_out = output_dir / "generation_samples.json"
        _save_json(gen_samples, gen_out)
        logger.info("  Generation samples saved: %s", gen_out)

    return results


# ===========================================================================
# Phase 2 — lm-eval Benchmarks (ORPO variant — already HF format)
# ===========================================================================

def _spawn_orpo_phase2_batch(
    hf_model_path: Path,
    output_dir: Path,
    gpu_task_list: list,
    num_fewshot: int,
    label_suffix: str,
    checkpoint: str,
    tokenizer: str,
) -> Dict[str, Any]:
    """Spawn Phase 2 subprocesses with ORPO environment."""
    processes = []

    for gpu_id, task_names, label in gpu_task_list:
        fewshot_label = f"[{num_fewshot}-shot] {label}"
        out_path = output_dir / f"phase2_gpu{gpu_id}_{num_fewshot}shot{label_suffix}.json"
        proc_info = _spawn_orpo_task(
            task_name="lm_eval",
            gpu_id=gpu_id,
            output_path=out_path,
            label=fewshot_label,
            checkpoint=checkpoint,
            tokenizer=tokenizer,
            extra_args={
                "--hf-model-path": str(hf_model_path),
                "--lm-eval-tasks": ",".join(task_names),
                "--num-fewshot": str(num_fewshot),
            },
        )
        processes.append(proc_info)

    return _wait_and_collect(processes)


def run_orpo_phase2(
    hf_model_path: Path,
    output_dir: Path,
    gpu_ids: List[int],
    checkpoint: str,
    tokenizer: str,
) -> Dict[str, Any]:
    """Run lm-eval benchmarks for ORPO model (0-shot + 5-shot)."""
    gpu_task_list = _build_phase2_tasks(gpu_ids)

    logger.info("  Running 0-shot benchmarks on %d GPUs ...", len(gpu_ids))
    results = _spawn_orpo_phase2_batch(
        hf_model_path, output_dir, gpu_task_list, 0, "",
        checkpoint, tokenizer,
    )
    logger.info("  Phase 2 (0-shot) complete.")

    # 5-shot
    logger.info("  Attempting 5-shot benchmarks ...")
    try:
        five_shot_results = _spawn_orpo_phase2_batch(
            hf_model_path, output_dir, gpu_task_list, 5, "_5shot",
            checkpoint, tokenizer,
        )
        logger.info("  Phase 2 (5-shot) complete.")
    except Exception:
        logger.warning("  5-shot failed (non-fatal): %s", traceback.format_exc())
        five_shot_results = {"error": traceback.format_exc()}
    results["5shot"] = five_shot_results

    phase2_out = output_dir / "phase2_results.json"
    _save_json(results, phase2_out)
    logger.info("  Phase 2 results saved: %s", phase2_out)
    return results


# ===========================================================================
# Phase 3 — 3-Way Comparison Report
# ===========================================================================

def run_orpo_phase3(
    phase1_results: Dict[str, Any],
    phase2_results: Dict[str, Any],
    output_dir: Path,
    base_results_dir: Path,
    sft_results_dir: Path,
    training_curve: Dict[str, Any],
    total_elapsed_sec: float,
) -> Optional[Path]:
    """Generate Base vs SFT vs ORPO 3-way comparison report."""
    try:
        from eval.report_generator import generate_three_way_report

        report_path = generate_three_way_report(
            base_results_dir=base_results_dir,
            sft_results_dir=sft_results_dir,
            orpo_phase1_results=phase1_results,
            orpo_phase2_results=phase2_results,
            output_path=_PROJECT_ROOT / "reports" / f"{datetime.now().strftime('%Y-%m-%d')}_ORPO_EVALUATION_REPORT.md",
            orpo_output_dir=output_dir,
            training_curve=training_curve,
            total_elapsed_sec=total_elapsed_sec,
        )
        logger.info("  3-way comparison report saved: %s", report_path)
        return report_path
    except Exception:
        logger.error("  Phase 3 report generation failed:\n%s", traceback.format_exc())

        # Fallback: dump raw JSON
        fallback = output_dir / "orpo_eval_summary.json"
        _save_json({
            "phase1": phase1_results,
            "phase2": phase2_results,
            "training_curve": training_curve,
        }, fallback)
        logger.info("  Fallback summary saved: %s", fallback)
        return None


# ===========================================================================
# CLI
# ===========================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="FRANKENSTALLM 3B — ORPO Evaluation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-phase1", action="store_true",
                        help="Skip internal eval.")
    parser.add_argument("--skip-phase2", action="store_true",
                        help="Skip lm-eval benchmarks.")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Override ORPO checkpoint path (auto-detects latest if not given).")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Override output directory.")
    parser.add_argument("--base-results", type=str, default=None,
                        help=f"Base eval results dir (default: {BASE_RESULTS_DIR})")
    parser.add_argument("--sft-results", type=str, default=None,
                        help=f"SFT eval results dir (default: {SFT_RESULTS_DIR})")
    parser.add_argument("--gpus", type=str, default=None,
                        help="Comma-separated GPU IDs (default: 0-7).")
    return parser.parse_args()


# ===========================================================================
# Main
# ===========================================================================

def main() -> None:
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    args = parse_args()

    # Resolve paths
    base_results_dir = Path(args.base_results) if args.base_results else BASE_RESULTS_DIR
    sft_results_dir = Path(args.sft_results) if args.sft_results else SFT_RESULTS_DIR

    # Auto-detect or use explicit checkpoint
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
    else:
        detected = detect_latest_checkpoint(ORPO_CHECKPOINT_DIR)
        if detected:
            checkpoint_path = detected
        else:
            logger.error(
                "No checkpoint-* subdirectory found under %s. "
                "Use --checkpoint to specify manually.",
                ORPO_CHECKPOINT_DIR,
            )
            sys.exit(1)

    checkpoint = str(checkpoint_path)
    tokenizer = resolve_tokenizer(checkpoint_path)

    # ORPO checkpoints are already in HF format (safetensors)
    hf_model_path = checkpoint_path

    # Output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        output_dir = _PROJECT_ROOT / "eval" / "outputs" / f"3b_orpo_eval_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # GPU IDs
    gpu_ids = sorted([int(g.strip()) for g in args.gpus.split(",")]) if args.gpus else list(range(8))

    # Dry run
    if args.dry_run:
        _print_banner("DRY RUN — ORPO Eval Pipeline")
        logger.info("  ORPO Checkpoint : %s", checkpoint)
        logger.info("  Tokenizer       : %s", tokenizer)
        logger.info("  HF Model Path   : %s (same as checkpoint)", hf_model_path)
        logger.info("  Base Results    : %s", base_results_dir)
        logger.info("  SFT Results     : %s", sft_results_dir)
        logger.info("  Output dir      : %s", output_dir)
        logger.info("  GPUs            : %s", gpu_ids)
        logger.info("  Chat template   : ENABLED for generation tasks")
        logger.info("")

        phase1_tasks = _build_phase1_tasks(gpu_ids)
        logger.info("  Phase 1 Tasks (%d):", len(phase1_tasks))
        for desc in phase1_tasks:
            is_gen = desc["task"] in ("generation", "repetition_grid")
            chat_mark = " [CHAT]" if is_gen else ""
            logger.info("    GPU %d — %s%s", desc["gpu_id"], desc["label"], chat_mark)

        phase2_tasks = _build_phase2_tasks(gpu_ids)
        logger.info("  Phase 2 Tasks (%d):", len(phase2_tasks))
        for gpu_id, tasks, label in phase2_tasks:
            logger.info("    GPU %d — %s", gpu_id, label)

        # Check Base results exist
        if base_results_dir.exists():
            p1_file = base_results_dir / "phase1_results.json"
            p2_file = base_results_dir / "phase2_results.json"
            logger.info("  Base phase1_results.json: %s", "OK" if p1_file.exists() else "MISSING")
            logger.info("  Base phase2_results.json: %s", "OK" if p2_file.exists() else "MISSING")
        else:
            logger.warning("  Base results dir NOT FOUND: %s", base_results_dir)

        # Check SFT results exist
        if sft_results_dir.exists():
            p1_file = sft_results_dir / "phase1_results.json"
            p2_file = sft_results_dir / "phase2_results.json"
            logger.info("  SFT  phase1_results.json: %s", "OK" if p1_file.exists() else "MISSING")
            logger.info("  SFT  phase2_results.json: %s", "OK" if p2_file.exists() else "MISSING")
        else:
            logger.warning("  SFT results dir NOT FOUND: %s", sft_results_dir)

        # Check train.log
        train_log = ORPO_CHECKPOINT_DIR / "train.log"
        logger.info("  train.log       : %s", "OK" if train_log.exists() else "MISSING")

        sys.exit(0)

    # -----------------------------------------------------------------------
    # Banner
    # -----------------------------------------------------------------------
    _print_banner("FRANKENSTALLM 3B — ORPO Evaluation Pipeline")
    logger.info("  ORPO Checkpoint : %s", checkpoint)
    logger.info("  Tokenizer       : %s", tokenizer)
    logger.info("  HF Model Path   : %s (same as checkpoint)", hf_model_path)
    logger.info("  Base Results    : %s", base_results_dir)
    logger.info("  SFT Results     : %s", sft_results_dir)
    logger.info("  Output dir      : %s", output_dir)
    logger.info("  GPUs            : %s", gpu_ids)
    logger.info("  Phases          : phase1=%s  phase2=%s",
                "skip" if args.skip_phase1 else "run",
                "skip" if args.skip_phase2 else "run")

    # Preflight checks
    if not Path(checkpoint).exists():
        logger.error("ORPO checkpoint not found: %s", checkpoint)
        sys.exit(1)
    if not Path(tokenizer).exists():
        logger.error("Tokenizer not found: %s", tokenizer)
        sys.exit(1)
    if not base_results_dir.exists():
        logger.warning("Base results dir not found: %s (Phase 3 may fail)", base_results_dir)
    if not sft_results_dir.exists():
        logger.warning("SFT results dir not found: %s (Phase 3 may fail)", sft_results_dir)
    logger.info("  Preflight OK: checkpoint=%s, tokenizer=%s", checkpoint, tokenizer)

    pipeline_start = time.time()
    phase1_results: Dict[str, Any] = {}
    phase2_results: Dict[str, Any] = {}

    # -----------------------------------------------------------------------
    # Extract training curve from train.log
    # -----------------------------------------------------------------------
    _print_phase_header("PRE-PHASE", "Extract Training Curve from train.log")
    train_log_path = ORPO_CHECKPOINT_DIR / "train.log"
    training_curve = extract_training_curve(train_log_path, output_dir)

    # -----------------------------------------------------------------------
    # Phase 1 — Internal Evaluation (8 GPU)
    # -----------------------------------------------------------------------
    _print_phase_header("PHASE 1", f"ORPO Internal Evaluation — {len(gpu_ids)} GPU Parallel")
    if args.skip_phase1:
        logger.info("  Skipping Phase 1.")
        phase1_out = output_dir / "phase1_results.json"
        if phase1_out.exists():
            with open(phase1_out, encoding="utf-8") as f:
                phase1_results = json.load(f)
            logger.info("  Loaded existing Phase 1 results.")
    else:
        t0 = time.time()
        try:
            phase1_results = run_orpo_phase1(output_dir, gpu_ids, checkpoint, tokenizer)
            logger.info("  Phase 1 complete in %s.", _fmt_seconds(time.time() - t0))
        except Exception:
            logger.error("  Phase 1 FAILED:\n%s", traceback.format_exc())

    # -----------------------------------------------------------------------
    # Phase 2 — lm-eval Benchmarks (8 GPU)
    # -----------------------------------------------------------------------
    _print_phase_header("PHASE 2", f"ORPO Benchmarks — {len(gpu_ids)} GPU Parallel")
    if args.skip_phase2:
        logger.info("  Skipping Phase 2.")
        phase2_out = output_dir / "phase2_results.json"
        if phase2_out.exists():
            with open(phase2_out, encoding="utf-8") as f:
                phase2_results = json.load(f)
            logger.info("  Loaded existing Phase 2 results.")
    else:
        t0 = time.time()
        try:
            phase2_results = run_orpo_phase2(
                hf_model_path, output_dir, gpu_ids, checkpoint, tokenizer,
            )
            logger.info("  Phase 2 complete in %s.", _fmt_seconds(time.time() - t0))
        except Exception:
            logger.error("  Phase 2 FAILED:\n%s", traceback.format_exc())

    # -----------------------------------------------------------------------
    # Phase 3 — 3-Way Comparison Report
    # -----------------------------------------------------------------------
    _print_phase_header("PHASE 3", "Base vs SFT vs ORPO — 3-Way Comparison Report")
    t0 = time.time()
    report_path = run_orpo_phase3(
        phase1_results, phase2_results, output_dir,
        base_results_dir, sft_results_dir,
        training_curve=training_curve,
        total_elapsed_sec=time.time() - pipeline_start,
    )
    logger.info("  Phase 3 complete in %s.", _fmt_seconds(time.time() - t0))

    # -----------------------------------------------------------------------
    # Final Summary
    # -----------------------------------------------------------------------
    total_elapsed = time.time() - pipeline_start
    _print_banner("ORPO EVALUATION PIPELINE COMPLETE")
    logger.info("  Total time       : %s", _fmt_seconds(total_elapsed))
    logger.info("  Output dir       : %s", output_dir)
    logger.info("  Training curve   : %s", output_dir / "training_curve.json")
    logger.info("  Phase 1 results  : %s", output_dir / "phase1_results.json")
    logger.info("  Phase 2 results  : %s", output_dir / "phase2_results.json")
    logger.info("  Report           : %s", report_path or "N/A")
    logger.info(_bar())


if __name__ == "__main__":
    main()
