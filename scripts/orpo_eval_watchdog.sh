#!/bin/bash
# =============================================================================
# ORPO Training Completion Watchdog
# =============================================================================
# Monitors the ORPO training process. When it finishes, automatically launches
# the comprehensive evaluation pipeline.
#
# Usage:
#   nohup bash scripts/orpo_eval_watchdog.sh > checkpoints/korean_3b_orpo_v1/watchdog.log 2>&1 &
# =============================================================================

set -euo pipefail

PROJECT_ROOT="/PROJECT/0325120031_A/ghong/taketimes/llm-bang"
TRAIN_LOG="${PROJECT_ROOT}/checkpoints/korean_3b_orpo_v1/train.log"
TRAIN_PID=$(pgrep -f "train/orpo.py.*korean_3b_orpo_v1" | head -1)

echo "=============================================="
echo "  ORPO Eval Watchdog Started"
echo "=============================================="
echo "  Time      : $(date '+%Y-%m-%d %H:%M:%S')"
echo "  Train PID : ${TRAIN_PID:-NOT FOUND}"
echo "  Train Log : ${TRAIN_LOG}"
echo "=============================================="

if [ -z "${TRAIN_PID}" ]; then
    echo "[WARN] Training process not found. Checking if already completed..."
    # Check if training already finished by looking for final output
    if grep -q "Training completed" "${TRAIN_LOG}" 2>/dev/null || \
       grep -q "Saving model checkpoint" "${TRAIN_LOG}" 2>/dev/null; then
        echo "[INFO] Training appears to have already completed."
    else
        echo "[ERROR] No training process and no completion marker found. Exiting."
        exit 1
    fi
else
    echo "[INFO] Watching training PID ${TRAIN_PID}..."
    echo ""

    # Poll every 60 seconds
    while kill -0 "${TRAIN_PID}" 2>/dev/null; do
        # Get current step
        CURRENT_STEP=$(grep -oP '\d+/9840' "${TRAIN_LOG}" 2>/dev/null | tail -1 || echo "?/?")
        LATEST_LOSS=$(grep "'loss':" "${TRAIN_LOG}" 2>/dev/null | tail -1 | grep -oP "'loss': '([^']+)'" | sed "s/'loss': '//;s/'//" || echo "?")
        echo "[$(date '+%H:%M:%S')] Step ${CURRENT_STEP} | Loss: ${LATEST_LOSS} | PID ${TRAIN_PID} running"
        sleep 60
    done

    echo ""
    echo "=============================================="
    echo "[INFO] Training process ${TRAIN_PID} has ended."
    echo "[INFO] Time: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "=============================================="
fi

# Wait a moment for any final I/O
sleep 10

# Get final training stats
echo ""
echo "[INFO] Final training stats:"
grep "eval_loss" "${TRAIN_LOG}" | tail -1 | tr ',' '\n' | head -10
echo ""

# Detect the latest checkpoint
LATEST_CKPT=$(ls -d ${PROJECT_ROOT}/checkpoints/korean_3b_orpo_v1/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1)
echo "[INFO] Latest checkpoint: ${LATEST_CKPT}"

if [ -z "${LATEST_CKPT}" ]; then
    echo "[ERROR] No checkpoint found. Cannot proceed with evaluation."
    exit 1
fi

# Send telegram notification (if available)
python3 -c "
import os, urllib.request, urllib.parse, json
token = os.environ.get('TELEGRAM_BOT_TOKEN', '')
chat_id = os.environ.get('TELEGRAM_CHAT_ID', '')
if token and chat_id:
    msg = '🏁 ORPO 학습 완료! 자동 평가 시작합니다.\nCheckpoint: ${LATEST_CKPT##*/}'
    url = f'https://api.telegram.org/bot{token}/sendMessage'
    data = urllib.parse.urlencode({'chat_id': chat_id, 'text': msg}).encode()
    urllib.request.urlopen(url, data, timeout=10)
    print('[INFO] Telegram notification sent.')
else:
    print('[INFO] Telegram not configured, skipping notification.')
" 2>/dev/null || true

# ============================================================================
# Launch evaluation pipeline
# ============================================================================
echo ""
echo "=============================================="
echo "  Starting ORPO Evaluation Pipeline"
echo "  Time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=============================================="

cd "${PROJECT_ROOT}"

python3 eval/orpo_eval_pipeline.py \
    --checkpoint "${LATEST_CKPT}" \
    2>&1 | tee -a checkpoints/korean_3b_orpo_v1/eval.log

EVAL_EXIT=$?

echo ""
echo "=============================================="
echo "  Evaluation Complete"
echo "  Exit code: ${EVAL_EXIT}"
echo "  Time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=============================================="

# Send completion notification
python3 -c "
import os, urllib.request, urllib.parse
token = os.environ.get('TELEGRAM_BOT_TOKEN', '')
chat_id = os.environ.get('TELEGRAM_CHAT_ID', '')
if token and chat_id:
    exit_code = ${EVAL_EXIT}
    status = '✅ 성공' if exit_code == 0 else '❌ 실패'
    msg = f'ORPO 평가 완료: {status}\nExit code: {exit_code}\n보고서: reports/ 확인'
    url = f'https://api.telegram.org/bot{token}/sendMessage'
    data = urllib.parse.urlencode({'chat_id': chat_id, 'text': msg}).encode()
    urllib.request.urlopen(url, data, timeout=10)
" 2>/dev/null || true

exit ${EVAL_EXIT}
