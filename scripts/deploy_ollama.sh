#!/usr/bin/env bash
# =============================================================================
# deploy_ollama.sh — FRANKENSTALLM 3B GGUF → Ollama 원클릭 배포
#
# Usage:
#   bash scripts/deploy_ollama.sh              # 기본 (Q4_K_M)
#   bash scripts/deploy_ollama.sh --quant Q8_0 # Q8_0 양자화
#   bash scripts/deploy_ollama.sh --skip_convert  # GGUF 이미 존재 시
#
# Pipeline:
#   1. [선택] GGUF 변환 + 양자화 (convert_3b_gguf.sh)
#   2. Ollama 설치 확인 / 서버 시작
#   3. Modelfile.3b로 모델 등록
#   4. 자동 테스트 (5개 프롬프트)
#   5. 반복률 검증 (15개 프롬프트)
# =============================================================================
set -euo pipefail

QUANT="${QUANT:-Q4_K_M}"
MODEL_NAME="frankenstallm-3b"
SKIP_CONVERT=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --quant)       QUANT="$2";        shift 2 ;;
        --skip_convert) SKIP_CONVERT=true; shift ;;
        -h|--help)
            grep '^#' "$0" | head -20 | sed 's/^# \{0,1\}//'
            exit 0 ;;
        *) echo "ERROR: 알 수 없는 옵션: $1"; exit 1 ;;
    esac
done

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

GGUF_PATH="outputs/gguf/frankenstallm-3b-${QUANT}.gguf"
MODELFILE="Modelfile.3b"

echo "=================================================================="
echo "  FRANKENSTALLM 3B Ollama 배포"
echo "  양자화  : $QUANT"
echo "  GGUF    : $GGUF_PATH"
echo "  Modelfile: $MODELFILE"
echo "=================================================================="

# ---- Step 1: GGUF 변환 (필요 시) ----
if [[ "$SKIP_CONVERT" == "false" ]]; then
    if [[ ! -f "$GGUF_PATH" ]]; then
        echo ""
        echo "[Step 1] GGUF 변환 실행 중 ..."
        bash scripts/convert_3b_gguf.sh \
            --input_dir checkpoints/korean_3b_orpo_v1/checkpoint-9840
    else
        echo "[Step 1] GGUF 파일 이미 존재 — 변환 건너뜀"
    fi
else
    echo "[Step 1] 변환 건너뜀 (--skip_convert)"
fi

if [[ ! -f "$GGUF_PATH" ]]; then
    echo "ERROR: GGUF 파일 없음: $GGUF_PATH"
    exit 1
fi

echo "  GGUF 크기: $(du -sh "$GGUF_PATH" | cut -f1)"

# ---- Step 2: Ollama 설치 확인 ----
if ! command -v ollama &>/dev/null; then
    echo ""
    echo "[Step 2] Ollama 미설치 — 설치 중 ..."
    curl -fsSL https://ollama.com/install.sh | sh
fi

# Ollama 서버 시작
if ! ollama list &>/dev/null 2>&1; then
    echo "[Step 2] Ollama 서버 시작 중 ..."
    ollama serve &>/tmp/ollama_serve.log &
    for i in $(seq 1 15); do
        if ollama list &>/dev/null 2>&1; then
            echo "  [OK] Ollama 서버 준비 (${i}초)"
            break
        fi
        sleep 1
    done
fi

# ---- Step 3: 모델 등록 ----
echo ""
echo "[Step 3] Ollama 모델 등록: $MODEL_NAME"
ollama create "$MODEL_NAME" -f "$MODELFILE"
echo "  [OK] 등록 완료"

# ---- Step 4: 자동 테스트 ----
echo ""
echo "[Step 4] 자동 테스트 ..."
declare -a QUICK_TESTS=(
    "대한민국의 수도는?"
    "인공지능이란 무엇인가요?"
    "한국의 전통 음식 중에서 김치에 대해 설명해주세요."
)

for prompt in "${QUICK_TESTS[@]}"; do
    echo "  Q: $prompt"
    RESP=$(timeout 60 ollama run "$MODEL_NAME" "$prompt" 2>&1 || echo "[TIMEOUT/ERROR]")
    echo "  A: ${RESP:0:200}"
    echo ""
done

# ---- Step 5: 반복률 검증 ----
echo "[Step 5] 반복률 검증 (15개 프롬프트) ..."
python3 scripts/test_ollama_repetition.py --model "$MODEL_NAME"

echo ""
echo "=================================================================="
echo "  배포 완료!"
echo "  사용법: ollama run $MODEL_NAME"
echo "=================================================================="
