"""
FRANKENSTALLM 3B — SFT Evaluation Pipeline Orchestrator
=========================================================

Evaluates the SFT checkpoint across 6 dimensions and generates a
comparison report against the Base model results.

Runs 4 phases sequentially:
  Phase 0  — Convert SFT checkpoint to HuggingFace format
  Phase 1  — Internal evaluation across 8 GPUs (PPL, Calibration, Generation)
  Phase 2  — Standard benchmarks via lm-eval-harness (8 GPU parallel)
  Phase 3  — Base vs SFT comparison report generation

Usage:
    python eval/sft_eval_pipeline.py
    python eval/sft_eval_pipeline.py --dry-run
    python eval/sft_eval_pipeline.py --skip-phase0 --skip-phase2
    python eval/sft_eval_pipeline.py --skip-phase0 --hf-model-path eval/outputs/hf_3b_sft_best
"""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
import os
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
# SFT checkpoint and Base results paths
# ---------------------------------------------------------------------------
SFT_CHECKPOINT = str(
    _PROJECT_ROOT / "checkpoints" / "korean_3b_sft_v2" / "checkpoint-best"
)
# SFT tokenizer lives alongside the SFT checkpoint
SFT_TOKENIZER = str(
    _PROJECT_ROOT / "checkpoints" / "korean_3b_sft_v2" / "tokenizer.json"
)
# Fallback tokenizer if SFT-specific one doesn't exist
_FALLBACK_TOKENIZER = str(
    _PROJECT_ROOT / "tokenizer" / "korean_sp" / "tokenizer.json"
)
BASE_RESULTS_DIR = _PROJECT_ROOT / "eval" / "outputs" / "3b_reeval_20260305_1451"

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
    run_phase0,
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
logger = logging.getLogger("sft_eval")


# ===========================================================================
# Override: spawn tasks with SFT environment variables
# ===========================================================================

def _spawn_sft_task(
    task_name: str,
    gpu_id: int,
    output_path: Path,
    label: str,
    checkpoint: str,
    tokenizer: str,
    use_chat_template: bool = False,
    extra_args: Optional[Dict[str, str]] = None,
) -> tuple:
    """Spawn a subprocess task with SFT checkpoint via environment variables."""
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

    logger.info("  Spawning: %s (GPU %d) [SFT]", label, gpu_id)
    proc = subprocess.Popen(
        cmd,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        env=env,
        cwd=str(_PROJECT_ROOT),
    )
    return proc, label, output_path, log_file


# ===========================================================================
# Phase 1 — Internal Evaluation (SFT variant)
# ===========================================================================

def run_sft_phase1(
    output_dir: Path,
    gpu_ids: List[int],
    checkpoint: str,
    tokenizer: str,
) -> Dict[str, Any]:
    """Run internal eval tasks with SFT checkpoint, chat template enabled for gen tasks."""
    task_descriptors = _build_phase1_tasks(gpu_ids)
    processes = []

    for desc in task_descriptors:
        is_gen_task = desc["task"] in ("generation", "repetition_grid")
        out_path = output_dir / f"phase1_{desc['task']}_gpu{desc['gpu_id']}.json"
        proc_info = _spawn_sft_task(
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
# Phase 2 — lm-eval Benchmarks (SFT variant — uses HF model)
# ===========================================================================

def _spawn_sft_phase2_batch(
    hf_model_path: Path,
    output_dir: Path,
    gpu_task_list: list,
    num_fewshot: int,
    label_suffix: str,
    checkpoint: str,
    tokenizer: str,
) -> Dict[str, Any]:
    """Spawn Phase 2 subprocesses with SFT environment."""
    processes = []

    for gpu_id, task_names, label in gpu_task_list:
        fewshot_label = f"[{num_fewshot}-shot] {label}"
        out_path = output_dir / f"phase2_gpu{gpu_id}_{num_fewshot}shot{label_suffix}.json"
        proc_info = _spawn_sft_task(
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


def run_sft_phase2(
    hf_model_path: Path,
    output_dir: Path,
    gpu_ids: List[int],
    checkpoint: str,
    tokenizer: str,
) -> Dict[str, Any]:
    """Run lm-eval benchmarks for SFT model (0-shot + 5-shot)."""
    gpu_task_list = _build_phase2_tasks(gpu_ids)

    logger.info("  Running 0-shot benchmarks on %d GPUs ...", len(gpu_ids))
    results = _spawn_sft_phase2_batch(
        hf_model_path, output_dir, gpu_task_list, 0, "",
        checkpoint, tokenizer,
    )
    logger.info("  Phase 2 (0-shot) complete.")

    # 5-shot
    logger.info("  Attempting 5-shot benchmarks ...")
    try:
        five_shot_results = _spawn_sft_phase2_batch(
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
# Phase 3 — Comparison Report
# ===========================================================================

def run_sft_phase3(
    phase1_results: Dict[str, Any],
    phase2_results: Dict[str, Any],
    output_dir: Path,
    base_results_dir: Path,
    total_elapsed_sec: float,
) -> Optional[Path]:
    """Generate Base vs SFT comparison report."""
    try:
        from eval.report_generator import generate_comparison_report

        report_path = generate_comparison_report(
            base_results_dir=base_results_dir,
            sft_phase1_results=phase1_results,
            sft_phase2_results=phase2_results,
            output_path=_PROJECT_ROOT / "reports" / f"{datetime.now().strftime('%Y-%m-%d')}_3B_SFT_V2_EVALUATION_REPORT.md",
            sft_output_dir=output_dir,
            total_elapsed_sec=total_elapsed_sec,
        )
        logger.info("  Comparison report saved: %s", report_path)
        return report_path
    except Exception:
        logger.error("  Phase 3 report generation failed:\n%s", traceback.format_exc())

        # Fallback: dump raw JSON
        fallback = output_dir / "sft_eval_summary.json"
        _save_json({
            "phase1": phase1_results,
            "phase2": phase2_results,
        }, fallback)
        logger.info("  Fallback summary saved: %s", fallback)
        return None


# ===========================================================================
# CLI
# ===========================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="FRANKENSTALLM 3B — SFT Evaluation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-phase0", action="store_true",
                        help="Skip HF conversion (reuse existing).")
    parser.add_argument("--skip-phase1", action="store_true",
                        help="Skip internal eval.")
    parser.add_argument("--skip-phase2", action="store_true",
                        help="Skip lm-eval benchmarks.")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help=f"Override SFT checkpoint (default: {SFT_CHECKPOINT})")
    parser.add_argument("--hf-model-path", type=str, default=None,
                        help="Pre-converted HF model path (skips Phase 0).")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Override output directory.")
    parser.add_argument("--base-results", type=str, default=None,
                        help=f"Base eval results dir (default: {BASE_RESULTS_DIR})")
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
    checkpoint = args.checkpoint or SFT_CHECKPOINT
    tokenizer = SFT_TOKENIZER if Path(SFT_TOKENIZER).exists() else _FALLBACK_TOKENIZER
    base_results_dir = Path(args.base_results) if args.base_results else BASE_RESULTS_DIR

    # Output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        output_dir = _PROJECT_ROOT / "eval" / "outputs" / f"3b_sft_eval_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # GPU IDs
    gpu_ids = sorted([int(g.strip()) for g in args.gpus.split(",")]) if args.gpus else list(range(8))

    # Dry run
    if args.dry_run:
        _print_banner("DRY RUN — SFT Eval Pipeline")
        logger.info("  SFT Checkpoint  : %s", checkpoint)
        logger.info("  Tokenizer       : %s", tokenizer)
        logger.info("  Base Results    : %s", base_results_dir)
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

        sys.exit(0)

    # -----------------------------------------------------------------------
    # Banner
    # -----------------------------------------------------------------------
    _print_banner("FRANKENSTALLM 3B — SFT Evaluation Pipeline")
    logger.info("  SFT Checkpoint  : %s", checkpoint)
    logger.info("  Tokenizer       : %s", tokenizer)
    logger.info("  Base Results    : %s", base_results_dir)
    logger.info("  Output dir      : %s", output_dir)
    logger.info("  GPUs            : %s", gpu_ids)
    logger.info("  Phases          : phase0=%s  phase1=%s  phase2=%s",
                "skip" if args.skip_phase0 else "run",
                "skip" if args.skip_phase1 else "run",
                "skip" if args.skip_phase2 else "run")

    # Preflight checks
    if not Path(checkpoint).exists():
        logger.error("SFT checkpoint not found: %s", checkpoint)
        sys.exit(1)
    if not Path(tokenizer).exists():
        logger.error("Tokenizer not found: %s", tokenizer)
        sys.exit(1)
    logger.info("  Preflight OK: checkpoint=%s, tokenizer=%s", checkpoint, tokenizer)

    pipeline_start = time.time()
    phase1_results: Dict[str, Any] = {}
    phase2_results: Dict[str, Any] = {}
    hf_model_path: Optional[Path] = None

    # -----------------------------------------------------------------------
    # Phase 0 — HF Conversion
    # -----------------------------------------------------------------------
    _print_phase_header("PHASE 0", "SFT Checkpoint → HuggingFace Conversion")
    if args.hf_model_path:
        hf_model_path = Path(args.hf_model_path)
        logger.info("  Using pre-converted HF model: %s", hf_model_path)
    elif args.skip_phase0:
        # Search for existing HF conversion
        candidate = output_dir / "hf_3b_sft_best"
        outputs_dir = _PROJECT_ROOT / "eval" / "outputs"
        if candidate.exists():
            hf_model_path = candidate
        else:
            candidates = list(outputs_dir.glob("hf_3b_sft*"))
            if candidates:
                hf_model_path = candidates[0]
        if hf_model_path:
            logger.info("  Skipping Phase 0 — reusing: %s", hf_model_path)
        else:
            logger.warning("  No HF model found. Phase 2 will be skipped.")
    else:
        t0 = time.time()
        try:
            hf_output = output_dir / "hf_3b_sft_best"
            hf_output.mkdir(parents=True, exist_ok=True)

            import subprocess
            convert_script = _PROJECT_ROOT / "scripts" / "convert_to_hf.py"
            cmd = [
                sys.executable, str(convert_script),
                "--checkpoint", checkpoint,
                "--output", str(hf_output),
                "--tokenizer", tokenizer,
            ]
            logger.info("  Running: %s", " ".join(cmd))
            subprocess.run(cmd, check=True, cwd=str(_PROJECT_ROOT))
            hf_model_path = hf_output
            logger.info("  Phase 0 complete in %s.", _fmt_seconds(time.time() - t0))
        except Exception:
            logger.error("  Phase 0 FAILED:\n%s", traceback.format_exc())

    # -----------------------------------------------------------------------
    # Phase 1 — Internal Evaluation (8 GPU)
    # -----------------------------------------------------------------------
    _print_phase_header("PHASE 1", f"SFT Internal Evaluation — {len(gpu_ids)} GPU Parallel")
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
            phase1_results = run_sft_phase1(output_dir, gpu_ids, checkpoint, tokenizer)
            logger.info("  Phase 1 complete in %s.", _fmt_seconds(time.time() - t0))
        except Exception:
            logger.error("  Phase 1 FAILED:\n%s", traceback.format_exc())

    # -----------------------------------------------------------------------
    # Phase 2 — lm-eval Benchmarks (8 GPU)
    # -----------------------------------------------------------------------
    _print_phase_header("PHASE 2", f"SFT Benchmarks — {len(gpu_ids)} GPU Parallel")
    if args.skip_phase2:
        logger.info("  Skipping Phase 2.")
        phase2_out = output_dir / "phase2_results.json"
        if phase2_out.exists():
            with open(phase2_out, encoding="utf-8") as f:
                phase2_results = json.load(f)
            logger.info("  Loaded existing Phase 2 results.")
    elif hf_model_path is None:
        logger.warning("  Phase 2 skipped — HF model unavailable.")
    else:
        t0 = time.time()
        try:
            phase2_results = run_sft_phase2(
                hf_model_path, output_dir, gpu_ids, checkpoint, tokenizer,
            )
            logger.info("  Phase 2 complete in %s.", _fmt_seconds(time.time() - t0))
        except Exception:
            logger.error("  Phase 2 FAILED:\n%s", traceback.format_exc())

    # -----------------------------------------------------------------------
    # Phase 3 — Comparison Report
    # -----------------------------------------------------------------------
    _print_phase_header("PHASE 3", "Base vs SFT Comparison Report")
    t0 = time.time()
    report_path = run_sft_phase3(
        phase1_results, phase2_results, output_dir, base_results_dir,
        total_elapsed_sec=time.time() - pipeline_start,
    )
    logger.info("  Phase 3 complete in %s.", _fmt_seconds(time.time() - t0))

    # -----------------------------------------------------------------------
    # Final Summary
    # -----------------------------------------------------------------------
    total_elapsed = time.time() - pipeline_start
    _print_banner("SFT EVALUATION PIPELINE COMPLETE")
    logger.info("  Total time      : %s", _fmt_seconds(total_elapsed))
    logger.info("  Output dir      : %s", output_dir)
    logger.info("  Phase 1 results : %s", output_dir / "phase1_results.json")
    logger.info("  Phase 2 results : %s", output_dir / "phase2_results.json")
    logger.info("  Report          : %s", report_path or "N/A")
    logger.info(_bar())


if __name__ == "__main__":
    main()
