#!/usr/bin/env bash
# =============================================================================
# launch_3b_sft_v2.sh — 8-GPU FP8 SFT v2 launcher for 3B Korean LLM
#
# SFT v2 improvements over v1:
#   - LR: 1e-5 → 5e-5 (5x, resolve underfitting)
#   - Effective batch: 64 → 256 (4x)
#   - Data mixing: 70% SFT + 30% pretrain (forgetting prevention)
#   - Weight decay: 0.01 → 0.05
#   - Warmup: 500 → 2000 steps
#   - Max steps: 33000 → 15000
#
# Usage:
#   bash scripts/launch_3b_sft_v2.sh
#   bash scripts/launch_3b_sft_v2.sh --max_steps 200    # quick test
#   bash scripts/launch_3b_sft_v2.sh --resume checkpoints/korean_3b_sft_v2/checkpoint-0002000
#
# Effective batch: 4 (local) x 8 GPU x 8 (grad_accum) = 256 samples/step
# =============================================================================
set -euo pipefail

# ---- Configurable defaults --------------------------------------------------
RUN_NAME="${RUN_NAME:-korean_3b_sft_v2}"
CONFIG="${CONFIG:-configs/korean_3b_sft_v2.yaml}"
BASE_CHECKPOINT="${BASE_CHECKPOINT:-checkpoints/korean_3b_fp8_run1/checkpoint-0057000}"
SFT_DATA="${SFT_DATA:-data/sft_combined/train_filtered.jsonl}"
VAL_DATA="${VAL_DATA:-data/sft_combined/val_filtered.jsonl}"
PRETRAIN_DATA="${PRETRAIN_DATA:-data/3b_train.bin}"
CKPT_DIR="checkpoints/${RUN_NAME}"
LOG_FILE="${CKPT_DIR}/train.log"
NPROC=8
MASTER_PORT="${MASTER_PORT:-29504}"

MAX_STEPS=15000
BATCH_SIZE=4
GRAD_ACCUM=8
LR="5.0e-5"
WARMUP_STEPS=2000
WEIGHT_DECAY=0.05
PRETRAIN_MIX_RATIO=0.3
SEED=42

EXTRA_ARGS="$@"

# ---- B200 / NVSwitch NCCL tuning (same as pretrain) -------------------------
export NCCL_IB_DISABLE=1
export NCCL_ALGO=Ring
export NCCL_PROTO=Simple
export NCCL_MIN_NCHANNELS=16
export NCCL_MAX_NCHANNELS=16
export NCCL_BUFFSIZE=67108864
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

# 3B + bs=4 VRAM allocation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "$(dirname "$0")/.."

# ---- Pre-flight checks ------------------------------------------------------
if [[ ! -d "${BASE_CHECKPOINT}" ]]; then
    echo "=================================================================="
    echo "  ERROR: Base checkpoint not found: ${BASE_CHECKPOINT}"
    echo "  Set BASE_CHECKPOINT env var or use --base_checkpoint CLI arg."
    echo "=================================================================="
    exit 1
fi

if [[ ! -f "${SFT_DATA}" ]]; then
    echo "=================================================================="
    echo "  ERROR: SFT data not found: ${SFT_DATA}"
    echo "  Run: bash scripts/prepare_sft_combined.sh"
    echo "=================================================================="
    exit 1
fi

if [[ ! -f "${PRETRAIN_DATA}" ]]; then
    echo "=================================================================="
    echo "  ERROR: Pretrain data not found: ${PRETRAIN_DATA}"
    echo "  Set PRETRAIN_DATA env var to the correct path."
    echo "=================================================================="
    exit 1
fi

# val fallback
if [[ ! -f "${VAL_DATA}" ]]; then
    VAL_FALLBACK="data/sft_combined/val.jsonl"
    if [[ -f "${VAL_FALLBACK}" ]]; then
        VAL_DATA="${VAL_FALLBACK}"
        echo "[INFO] val_filtered not found, fallback: ${VAL_DATA}"
    else
        echo "ERROR: VAL_DATA not found: ${VAL_DATA}"
        exit 1
    fi
fi

mkdir -p "${CKPT_DIR}"

echo "=================================================================="
echo "  3B SFT v2 Fine-Tuning"
echo "  Run name        : ${RUN_NAME}"
echo "  Config          : ${CONFIG}"
echo "  Base checkpoint : ${BASE_CHECKPOINT}"
echo "  SFT data        : ${SFT_DATA}"
echo "  Pretrain data   : ${PRETRAIN_DATA}"
echo "  Val data        : ${VAL_DATA}"
echo "  CKPT dir        : ${CKPT_DIR}"
echo "  Log file        : ${LOG_FILE}"
echo "  Max steps       : ${MAX_STEPS}"
echo "  Batch size      : ${BATCH_SIZE} (local) x ${NPROC} GPU x ${GRAD_ACCUM} grad_accum = $((BATCH_SIZE * NPROC * GRAD_ACCUM)) eff_batch"
echo "  Learning rate   : ${LR}"
echo "  Weight decay    : ${WEIGHT_DECAY}"
echo "  Warmup          : ${WARMUP_STEPS} steps"
echo "  Data mixing     : $((100 - ${PRETRAIN_MIX_RATIO%.*}0))% SFT + ${PRETRAIN_MIX_RATIO}00% pretrain"
echo "  Master port     : ${MASTER_PORT}"
echo "  ALLOC_CONF      : ${PYTORCH_CUDA_ALLOC_CONF}"
echo "  Started         : $(date)"
echo "=================================================================="

export PYTHONWARNINGS="ignore::UserWarning:torch.library"

torchrun \
    --nproc_per_node=${NPROC} \
    --master_port=${MASTER_PORT} \
    train/sft.py \
    --config "${CONFIG}" \
    --base_checkpoint "${BASE_CHECKPOINT}" \
    --sft_data "${SFT_DATA}" \
    --val_data "${VAL_DATA}" \
    --pretrain_data "${PRETRAIN_DATA}" \
    --pretrain_mix_ratio ${PRETRAIN_MIX_RATIO} \
    --checkpoint_dir "${CKPT_DIR}" \
    --log_file "${LOG_FILE}" \
    --max_steps ${MAX_STEPS} \
    --batch_size ${BATCH_SIZE} \
    --grad_accum ${GRAD_ACCUM} \
    --lr ${LR} \
    --weight_decay ${WEIGHT_DECAY} \
    --warmup_steps ${WARMUP_STEPS} \
    --seed ${SEED} \
    --use_fp8 \
    ${EXTRA_ARGS} \
    2>&1 | grep -v "UserWarning" \
         | grep -v "Warning only once" \
         | grep -v "Overriding a previously" \
         | grep -v "dispatch key:" \
         | grep -v "previous kernel:" \
         | grep -v "new kernel:" \
         | grep -v "operator: flash_attn" \
         | grep -v "registered at /usr/local" \
         | grep -v "self.m.impl" \
         | tee -a "${LOG_FILE}"

echo "=================================================================="
echo "  3B SFT v2 Done : $(date)"
echo "=================================================================="
