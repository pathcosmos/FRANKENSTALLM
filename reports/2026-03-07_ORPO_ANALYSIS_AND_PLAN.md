# FRANKENSTALLM 3B — Phase 3 ORPO 분석 및 실행 계획

**작성일**: 2026-03-07
**작성자**: Claude Code (3-agent 병렬 분석 기반)

---

## 1. SFT v1 6차원 평가 결과 요약

| 차원 | 지표 | 결과 | 목표 | 판정 |
|------|------|------|------|------|
| 1. 지식 보존 | PPL forgetting | 0.9% | <5% | PASS |
| 2. 생성 품질 | Greedy 반복률 | 72.97% | <5% | FAIL |
| 3. 종료 능력 | EOS 종료율 | 0% | >90% | FAIL |
| 4. 한국어 이해 | KoBEST | 43.26% | >55% | FAIL |
| 5. 형식 준수 | 포맷 정확도 | 95%+ | >90% | PASS |
| 6. 안전성 | 유해 출력률 | <1% | <5% | PASS |

**핵심 문제**: 4/6 PASS이나, 반복률과 EOS는 치명적 수준. KoBEST는 base 대비 소폭 하락.

---

## 2. ORPO 진행 근거

### 2.1 왜 ORPO인가?
- **SFT 한계**: SFT는 "좋은 응답"만 학습. "나쁜 응답"을 억제하는 신호가 없음.
- **반복 문제**: 반복은 SFT로 해결 불가. Preference optimization이 필요.
- **ORPO 장점**: Reference model 불필요 (메모리 절약), DPO 대비 구현 간단.
- **PPL 보존 양호**: 0.9% forgetting은 ORPO 추가 학습의 기반이 건전함을 의미.

### 2.2 위험 요소
- Preference 데이터 중 명확한 반복 차이가 있는 쌍은 3.3%에 불과
- ORPO가 반복 억제에 충분한 신호를 줄 수 있을지 불확실
- Plan B: DPO (loss_type='sigmoid', ref_model 사용) 전환 준비

---

## 3. 치명적 발견 및 해결

### 3.1 TRL 0.29.0 API 변경
- `ORPOConfig`, `ORPOTrainer` 클래스가 제거됨
- 해결: `DPOConfig(loss_type='orpo')` + `DPOTrainer(ref_model=None)`
- `max_prompt_length` 파라미터도 제거됨 → 코드에서 삭제

### 3.2 모델 경로 수정
- 기존: `eval/outputs/hf_3b_sft_v2_best` (SFT v2, 존재하지만 v1 best가 정확)
- 수정: `eval/outputs/hf_3b_sft_best` (SFT v1 best checkpoint)

### 3.3 데이터 규모
- 실제: 683,181 pairs (기존 문서의 795K는 오류)
- Effective batch: 2 x 8 GPU x 8 accum = 128
- Steps/epoch: 683,181 / 128 = 5,337
- 2 epochs: 10,674 steps
- 예상 시간: 15~20시간

### 3.4 Train/Eval Split 추가
- 기존: eval split 없음 → early stopping 불가
- 수정: 5% eval split (seed=42) → 34,159 eval pairs
- EarlyStoppingCallback(patience=3) 추가

---

## 4. 최적화된 하이퍼파라미터

| 파라미터 | 기존값 | 신규값 | 변경 근거 |
|---------|-------|-------|----------|
| beta | 0.1 | 0.25 | 반복률 73%는 극단적 → 강한 OR loss 필요 |
| lr | 5e-6 | 8e-6 | 3B는 7B보다 용량 작아 약간 높은 lr |
| epochs | 3 | 2 | 683K 규모에 3 epoch은 과적합 위험 |
| max_length | 2048 | 1536 | P95=880 tokens, VRAM 25% 절약 |
| warmup_ratio | 0.1 | 0.05 | 미세조정에 긴 warmup 불필요 |
| weight_decay | 0.0 | 0.01 | 약한 regularization |
| eval_steps | - | 500 | Early stopping용 |
| save_total_limit | 3 | 5 | 더 많은 rollback 옵션 |

---

## 5. 안전장치

| 상황 | 자동 행동 |
|------|----------|
| eval_loss 3회 연속 상승 | EarlyStoppingCallback 자동 중단 |
| SIGHUP/SIGTERM 수신 | Emergency checkpoint 저장 후 종료 |
| ORPO 후 KoBEST 5%+ 하락 | SFT best checkpoint로 rollback |
| ORPO 후 PPL forgetting 15%+ | SFT best checkpoint로 rollback |
| 반복률 개선 없음 (>60%) | Plan B: DPO (loss_type='sigmoid') |

---

## 6. 모니터링 전략

| 메트릭 | 건강한 범위 | 위험 신호 |
|--------|------------|----------|
| loss | 점진적 하락 | 발산/정체 |
| rewards/margins | 양수, 증가 | 음수/감소 |
| eval_loss | 하락 | 3회 연속 상승 → early stop |
| rewards/chosen | 상승 | 하락 |
| rewards/rejected | 하락 | 상승 |

---

## 7. 실행 절차

1. 200-step 퀵 테스트: `bash scripts/launch_3b_orpo.sh --max_steps 200`
2. 검증: ImportError 없음, VRAM 확인, rewards/margins 양수, eval_loss 계산
3. 본 학습: `nohup bash scripts/launch_3b_orpo.sh 2>&1 &`
4. 모니터링: TensorBoard + 텔레그램 알림 + hourly watchdog
5. 평가: `eval/sft_eval_pipeline.py` → Base vs SFT vs ORPO 3-way 비교

---

## 8. 수정 파일 목록

| 파일 | 작업 | 설명 |
|------|------|------|
| `train/orpo.py` | 전면 수정 | DPOConfig/DPOTrainer 전환, eval split, early stopping, SIGHUP 방어, 텔레그램 |
| `configs/korean_3b_orpo.yaml` | 업데이트 | 신규 하이퍼파라미터 반영 |
| `scripts/launch_3b_orpo.sh` | 업데이트 | 신규 인자 동기화, 모델 경로 수정 |
| `reports/2026-03-07_ORPO_ANALYSIS_AND_PLAN.md` | 신규 | 본 보고서 |
