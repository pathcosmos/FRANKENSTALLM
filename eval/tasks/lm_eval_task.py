"""
lm_eval_task.py — lm-evaluation-harness integration task.

Top-level function for ProcessPoolExecutor (spawn) compatibility:
  - run_lm_eval_tasks(hf_model_path, tasks, device, num_fewshot=0) -> dict

Requires: lm_eval >= 0.4 (installed as lm-eval 0.4.11)
"""
from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

CHECKPOINT = str(_PROJECT_ROOT / "checkpoints" / "korean_3b_fp8_run1" / "checkpoint-0057000")
TOKENIZER_PATH = str(_PROJECT_ROOT / "tokenizer" / "korean_sp" / "tokenizer.json")
DATA_DIR = _PROJECT_ROOT / "data"
SEQ_LEN = 2048
STRIDE = 512
BATCH_SIZE = 32

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Main task function (must be top-level for pickle / spawn compatibility)
# ---------------------------------------------------------------------------

def run_lm_eval_tasks(
    hf_model_path: str,
    tasks: list[str],
    device: str,
    num_fewshot: int = 0,
) -> dict:
    """Run lm-evaluation-harness benchmarks on a HuggingFace-format model.

    Isolates a single GPU via CUDA_VISIBLE_DEVICES so the function is safe
    to run in a ProcessPoolExecutor worker without VRAM conflicts.

    Args:
        hf_model_path: Path to a HuggingFace-compatible model directory
                       (must contain config.json + safetensors/pytorch_model).
        tasks:         List of lm-eval task names, e.g.
                       ["hellaswag", "arc_easy", "piqa"].
                       Unknown tasks are skipped with a warning.
        device:        CUDA device string, e.g. "cuda:7".
                       The function maps this to CUDA_VISIBLE_DEVICES=7 and
                       then uses device="cuda:0" inside lm_eval.
        num_fewshot:   Number of few-shot examples (0 = zero-shot).

    Returns:
        Dict with keys:
          - model_path:     hf_model_path as provided
          - tasks_requested: original task list
          - tasks_evaluated: tasks that were actually run
          - tasks_skipped:   tasks that were not available / errored
          - per_task_metrics: dict mapping task name to metric sub-dict
          - raw_results:     full results dict from lm_eval.simple_evaluate
          - elapsed_sec:     wall-clock time for the evaluation
    """
    # --- GPU isolation ---
    gpu_index = int(device.split(":")[-1])
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
    # After this point use cuda:0 since only one GPU is visible
    _internal_device = "cuda:0"

    print(
        f"[LM_EVAL] Starting on {device} "
        f"(CUDA_VISIBLE_DEVICES={gpu_index}), tasks={tasks}, "
        f"num_fewshot={num_fewshot}"
    )

    # --- Validate task list ---
    try:
        import lm_eval  # type: ignore[import]
        from lm_eval.tasks import TaskManager  # type: ignore[import]

        task_manager = TaskManager()
        available_tasks: set[str] = set(task_manager.all_tasks)
    except Exception as exc:
        logger.warning(f"[LM_EVAL] Could not enumerate available tasks: {exc}")
        available_tasks = set()  # will attempt all and catch errors per task

    valid_tasks: list[str] = []
    skipped_tasks: list[str] = []

    for t in tasks:
        if (not available_tasks) or (t in available_tasks):
            valid_tasks.append(t)
        else:
            logger.warning(f"[LM_EVAL] Task '{t}' not found in lm_eval registry — skipping.")
            skipped_tasks.append(t)

    if not valid_tasks:
        print("[LM_EVAL] No valid tasks to evaluate.")
        return {
            "model_path": hf_model_path,
            "tasks_requested": tasks,
            "tasks_evaluated": [],
            "tasks_skipped": skipped_tasks,
            "per_task_metrics": {},
            "raw_results": {},
            "elapsed_sec": 0.0,
        }

    # --- Run evaluation ---
    t0 = time.time()
    raw_results: dict[str, Any] = {}
    evaluated_tasks: list[str] = []
    error_tasks: list[str] = []

    # Attempt all valid tasks together first; fall back to per-task on error
    try:
        print(
            f"[LM_EVAL] Evaluating {len(valid_tasks)} task(s) together: {valid_tasks}"
        )
        raw_results = lm_eval.simple_evaluate(
            model="hf",
            model_args=(
                f"pretrained={hf_model_path},"
                f"dtype=bfloat16,"
                f"device={_internal_device}"
            ),
            tasks=valid_tasks,
            num_fewshot=num_fewshot,
            batch_size="auto",
        )
        evaluated_tasks = list(valid_tasks)

    except Exception as exc:
        logger.warning(
            f"[LM_EVAL] Batch evaluation failed ({exc}). "
            "Falling back to per-task evaluation."
        )
        # Fall back: evaluate one task at a time
        for task_name in valid_tasks:
            try:
                print(f"[LM_EVAL] Evaluating task '{task_name}' individually...")
                task_result = lm_eval.simple_evaluate(
                    model="hf",
                    model_args=(
                        f"pretrained={hf_model_path},"
                        f"dtype=bfloat16,"
                        f"device={_internal_device}"
                    ),
                    tasks=[task_name],
                    num_fewshot=num_fewshot,
                    batch_size="auto",
                    device=_internal_device,
                )
                # Merge per-task results into raw_results
                if not raw_results:
                    raw_results = task_result
                else:
                    if "results" in task_result and "results" in raw_results:
                        raw_results["results"].update(task_result.get("results", {}))
                evaluated_tasks.append(task_name)
            except Exception as task_exc:
                logger.warning(
                    f"[LM_EVAL] Task '{task_name}' failed: {task_exc}"
                )
                error_tasks.append(task_name)

    skipped_tasks.extend(error_tasks)
    elapsed = time.time() - t0

    # --- Extract per-task metrics ---
    per_task_metrics: dict[str, dict] = {}
    lm_results: dict[str, Any] = raw_results.get("results", {})

    for task_name in evaluated_tasks:
        if task_name in lm_results:
            task_data = lm_results[task_name]
            metrics: dict[str, Any] = {}
            for key, value in task_data.items():
                # Skip non-metric metadata keys
                if key in ("alias", "group"):
                    continue
                metrics[key] = value
            per_task_metrics[task_name] = metrics
        else:
            logger.warning(
                f"[LM_EVAL] Task '{task_name}' not found in results dict after evaluation."
            )

    # --- Summary print ---
    print(f"[LM_EVAL] Evaluation complete in {elapsed:.1f}s")
    for task_name, metrics in per_task_metrics.items():
        # Print the most common accuracy variants
        acc = metrics.get("acc,none") or metrics.get("acc") or metrics.get("accuracy")
        acc_norm = metrics.get("acc_norm,none") or metrics.get("acc_norm")
        if acc is not None:
            line = f"  {task_name}: acc={acc:.4f}"
            if acc_norm is not None:
                line += f", acc_norm={acc_norm:.4f}"
            print(f"[LM_EVAL] {line}")
        else:
            print(f"[LM_EVAL]   {task_name}: {metrics}")

    if skipped_tasks:
        print(f"[LM_EVAL] Skipped tasks: {skipped_tasks}")

    return {
        "model_path": hf_model_path,
        "tasks_requested": tasks,
        "tasks_evaluated": evaluated_tasks,
        "tasks_skipped": skipped_tasks,
        "per_task_metrics": per_task_metrics,
        "raw_results": raw_results,
        "elapsed_sec": round(elapsed, 1),
    }
