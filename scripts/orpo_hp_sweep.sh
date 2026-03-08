#!/usr/bin/env bash
# =============================================================================
# orpo_hp_sweep.sh — ORPO Hyperparameter Sweep (200 steps each)
#
# 각 설정을 200 steps씩 돌려서 최적 조합을 찾는 스크립트.
# 결과는 sweep_results/ 디렉토리에 저장됨.
#
# Usage:
#   bash scripts/orpo_hp_sweep.sh           # 전체 sweep (6 runs)
#   bash scripts/orpo_hp_sweep.sh --dry-run # 설정만 출력
# =============================================================================
set -uo pipefail
# NOTE: set +e — individual runs may fail; we log failures and continue the sweep

cd "$(dirname "$0")/.."

SWEEP_STEPS=200
SWEEP_DIR="checkpoints/orpo_sweep"
RESULTS_FILE="${SWEEP_DIR}/sweep_results.jsonl"
BASE_MODEL="eval/outputs/hf_3b_sft_best"
DATA_PATH="data/preference/combined_preference.jsonl"
NPROC=8
MASTER_PORT_BASE=29510

# B200 NCCL tuning (NVSwitch mesh — let NCCL auto-detect proto/channels/algo)
export NCCL_IB_DISABLE=1
export NCCL_BUFFSIZE=134217728
export OMP_NUM_THREADS=9
export MKL_NUM_THREADS=9
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_P2P_LEVEL=NVL
export PYTHONWARNINGS="ignore::UserWarning:torch.library"

mkdir -p "${SWEEP_DIR}"
declare -a FAILED_RUNS=()

# ---------------------------------------------------------------------------
# Sweep configurations: (name, beta, lr, max_length, batch_size, grad_accum)
# ---------------------------------------------------------------------------
# 핵심 탐색 축:
#   1. beta: 반복 억제 강도 (0.15 vs 0.25 vs 0.35)
#   2. lr: 수렴 속도 (5e-6 vs 8e-6 vs 1.2e-5)
#   3. max_length: VRAM vs 커버리지 (1024 vs 1536)

declare -a CONFIGS=(
    # name                beta   lr      max_len bs  accum
    "baseline_b015_lr8e6  0.15   8e-6    1536    4   4"
    "baseline_b025_lr8e6  0.25   8e-6    1536    4   4"
    "strong_b035_lr8e6    0.35   8e-6    1536    4   4"
    "fast_b025_lr12e6     0.25   1.2e-5  1536    4   4"
    "conserv_b025_lr5e6   0.25   5e-6    1536    4   4"
    "short_b025_lr8e6     0.25   8e-6    1024    4   4"
)

DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=true
fi

echo "=================================================================="
echo "  ORPO Hyperparameter Sweep"
echo "  Configs: ${#CONFIGS[@]}"
echo "  Steps each: ${SWEEP_STEPS}"
echo "  Results: ${RESULTS_FILE}"
echo "=================================================================="

for i in "${!CONFIGS[@]}"; do
    read -r NAME BETA LR MAX_LEN BS ACCUM <<< "${CONFIGS[$i]}"
    PORT=$((MASTER_PORT_BASE + i))
    OUTPUT="${SWEEP_DIR}/${NAME}"

    echo ""
    echo "--- Run $((i+1))/${#CONFIGS[@]}: ${NAME} ---"
    echo "    beta=${BETA} lr=${LR} max_length=${MAX_LEN} bs=${BS} accum=${ACCUM}"

    if [[ "${DRY_RUN}" == "true" ]]; then
        echo "    [DRY RUN] skipping"
        continue
    fi

    mkdir -p "${OUTPUT}"
    START_TIME=$(date +%s)

    torchrun \
        --nproc_per_node=${NPROC} \
        --master_port=${PORT} \
        train/orpo.py \
        --model_path "${BASE_MODEL}" \
        --custom_data_path "${DATA_PATH}" \
        --output_dir "${OUTPUT}" \
        --max_steps ${SWEEP_STEPS} \
        --lr ${LR} \
        --beta ${BETA} \
        --batch_size ${BS} \
        --gradient_accumulation_steps ${ACCUM} \
        --max_length ${MAX_LEN} \
        \
        --weight_decay 0.01 \
        --warmup_ratio 0.05 \
        --eval_split_ratio 0.05 \
        --eval_steps 100 \
        --early_stopping_patience 100 \
        --save_steps 200 \
        --save_total_limit 1 \
        --logging_steps 10 \
        --report_to none \
        --dataset_num_proc 64 \
        --dataloader_num_workers 4 \
        --no_load_best \
        2>&1 | tee "${OUTPUT}/train.log"
    RUN_EXIT=$?

    END_TIME=$(date +%s)
    ELAPSED=$((END_TIME - START_TIME))

    if [[ ${RUN_EXIT} -ne 0 ]]; then
        echo "    [ERROR] Run ${NAME} failed with exit code ${RUN_EXIT} after ${ELAPSED}s"
        echo "{\"name\":\"${NAME}\",\"beta\":${BETA},\"lr\":\"${LR}\",\"max_length\":${MAX_LEN},\"status\":\"FAILED\",\"exit_code\":${RUN_EXIT},\"elapsed_s\":${ELAPSED}}" >> "${RESULTS_FILE}"
        FAILED_RUNS+=("${NAME}")
        continue
    fi

    # Extract final metrics from log
    FINAL_LOSS=$(grep -oP "'loss': '[\d.]+'" "${OUTPUT}/train.log" | tail -1 | grep -oP "[\d.]+" || echo "N/A")
    EVAL_LOSS=$(grep -oP "'eval_loss': '[\d.]+'" "${OUTPUT}/train.log" | tail -1 | grep -oP "[\d.]+" || echo "N/A")
    MARGIN=$(grep -oP "'rewards/margins': '[-\d.]+'" "${OUTPUT}/train.log" | tail -1 | grep -oP "[-\d.]+" || echo "N/A")

    # Save result
    echo "{\"name\":\"${NAME}\",\"beta\":${BETA},\"lr\":\"${LR}\",\"max_length\":${MAX_LEN},\"status\":\"OK\",\"loss\":\"${FINAL_LOSS}\",\"eval_loss\":\"${EVAL_LOSS}\",\"margin\":\"${MARGIN}\",\"elapsed_s\":${ELAPSED}}" >> "${RESULTS_FILE}"

    echo "    -> loss=${FINAL_LOSS} eval_loss=${EVAL_LOSS} margin=${MARGIN} time=${ELAPSED}s"

    # Cleanup weights to save disk (keep logs)
    rm -rf "${OUTPUT}/checkpoint-"* "${OUTPUT}/emergency_checkpoint" 2>/dev/null || true
done

echo ""
echo "=================================================================="
echo "  Sweep Complete!"
echo "  Results: ${RESULTS_FILE}"
if [[ -f "${RESULTS_FILE}" ]]; then
    echo ""
    echo "  Summary:"
    cat "${RESULTS_FILE}" | python3 -c "
import sys, json
results = [json.loads(l) for l in sys.stdin]
results.sort(key=lambda r: float(r.get('eval_loss', '999')))
print(f'  {\"Name\":<25} {\"Beta\":>6} {\"LR\":>10} {\"Loss\":>8} {\"EvalLoss\":>10} {\"Margin\":>8} {\"Time\":>6}')
print(f'  {\"-\"*25} {\"-\"*6} {\"-\"*10} {\"-\"*8} {\"-\"*10} {\"-\"*8} {\"-\"*6}')
for r in results:
    print(f'  {r[\"name\"]:<25} {r[\"beta\"]:>6} {r[\"lr\"]:>10} {r[\"loss\"]:>8} {r[\"eval_loss\"]:>10} {r[\"margin\"]:>8} {r[\"elapsed_s\"]:>5}s')
print()
best = results[0]
print(f'  BEST: {best[\"name\"]} (eval_loss={best[\"eval_loss\"]})')
" 2>/dev/null || cat "${RESULTS_FILE}"
fi

# Report failed runs
if [[ ${#FAILED_RUNS[@]} -gt 0 ]]; then
    echo ""
    echo "  FAILED RUNS (${#FAILED_RUNS[@]}):"
    for fname in "${FAILED_RUNS[@]}"; do
        echo "    - ${fname}"
    done
fi
echo "=================================================================="
