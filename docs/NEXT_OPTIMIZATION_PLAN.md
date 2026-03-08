# 다음 프리트레인 최적화 계획 — MFU 33.5% → 47% 목표

> 작성일: 2026-03-08
> Phase 3 ORPO 완료 후, 다음 프리트레인 런 전에 적용할 최적화 항목을 정리한다.

---

## 1. 현재 성능 진단

### 1.1 Phase 1 실측 수치

| 항목 | 값 |
|------|-----|
| Steps | 57,000 |
| 토큰 | ~38.5B |
| 소요 시간 | 약 63시간 |
| GPU당 처리 속도 | 36~38K tok/s |
| 전체 처리 속도 (8GPU) | ~292K tok/s |
| MFU | **~33.5%** |

### 1.2 MFU가 낮은 근본 원인

README 8섹션에 이미 기록된 미적용 최적화 항목들:

- **QKV fusion** (+8~12%)
- **NUMA affinity** (+4~9%) — "5/8 rank가 잘못된 NUMA 노드" 언급
- **FA2 native RoPE** (+3~5%)

**결정적 문제:**

- 초기 DDP 런칭 시 **5/8 rank가 잘못된 NUMA 노드에서 실행**
- **69%의 DataLoader worker가 크로스-NUMA**
- NUMA affinity 최적화는 **미적용 상태** (로드맵 항목)

---

## 2. 수치로 계산한 손실분

```
B200 × 8 이론 FP8: 18,000 TFLOPS
MFU 33.5% 실효:    6,030 TFLOPS

만약 최적화했다면:
  QKV fusion +10%:   → 36.8% MFU
  NUMA affinity +6%: → 42.8% MFU  ← 가장 큰 손실
  FA2 RoPE +4%:      → 46.8% MFU

이론적 달성 가능 MFU: ~47%
실효 처리량: 8,460 TFLOPS

현재 292K tok/s → 최적화 시 약 410K tok/s
63시간 → 약 45시간으로 단축 가능했음
```

---

## 3. NUMA가 얼마나 치명적인가

```
AMD EPYC 9365 × 2소켓 구성:
  - GPU 0~3 → NUMA node 0 (core 0-35)
  - GPU 4~7 → NUMA node 1 (core 36-71)

69%의 DataLoader worker가 크로스-NUMA
= 잘못된 소켓의 CPU가 데이터를 GPU에 밀어넣는 상황

크로스-NUMA PCIe 전송은 같은 NUMA 대비 ~2배 지연
→ GPU가 데이터 기다리는 idle time 발생
→ MFU 직접 하락
```

---

## 4. 적용 가능한 최적화

### 4.1 NUMA affinity 고정 (가장 큰 효과, +4~9%)

**방법 A — launch script 수정:**

```bash
# launch_3b_pretrain.sh 수정
# GPU 0-3은 NUMA 0, GPU 4-7은 NUMA 1에 바인딩

numactl --cpunodebind=0 --membind=0 torchrun \
  --nproc_per_node=4 \
  --node_rank=0 \
  train/pretrain.py ... &

numactl --cpunodebind=1 --membind=1 torchrun \
  --nproc_per_node=4 \
  --node_rank=1 \
  train/pretrain.py ... &
```

**방법 B — Python 내 affinity 설정:**

```python
# train/pretrain.py
import os
local_rank = int(os.environ["LOCAL_RANK"])
numa_node = 0 if local_rank < 4 else 1
os.sched_setaffinity(0, range(numa_node*36, (numa_node+1)*36))
```

### 4.2 QKV Fusion — TransformerEngine 활용 (+8~12%)

```python
# model/attention.py
# 현재: Q, K, V 별도 linear
# 변경: te.Linear로 QKV 합산 후 split

import transformer_engine.pytorch as te

# Before
self.q_proj = te.Linear(d_model, d_model)
self.k_proj = te.Linear(d_model, n_kv_heads * head_dim)
self.v_proj = te.Linear(d_model, n_kv_heads * head_dim)

# After — 단일 kernel로 QKV 동시 처리
self.qkv_proj = te.Linear(
    d_model,
    d_model + 2 * n_kv_heads * head_dim,
    bias=False
)
```

### 4.3 DataLoader NUMA 정렬 (+3~5%)

```python
# train/pretrain.py DataLoader 설정
DataLoader(
    dataset,
    num_workers=8,          # 현재값 확인 필요
    pin_memory=True,
    pin_memory_device=f"cuda:{local_rank}",  # 추가
    persistent_workers=True  # worker 재생성 오버헤드 제거
)
```

### 4.4 NCCL 환경변수 추가

```bash
# scripts/launch_3b_pretrain.sh에 추가
export NCCL_MIN_NCHANNELS=4    # NVLink 채널 최소값 보장
export NCCL_SOCKET_NTHREADS=4  # 소켓 스레드 증가
export CUDA_DEVICE_MAX_CONNECTIONS=1  # 통신-연산 오버랩
```

### 4.5 FA2 native RoPE (+3~5%)

FA2에서 RoPE를 native로 적용하면 별도 RoPE 커널 호출을 제거할 수 있음. 구체적 구현은 FA2 버전(`2.7.4.post1+25.12`)에 따라 확인 필요.

---

## 5. 최적화 전후 예상 비교

| 항목 | 현재 | 최적화 후 예상 |
|------|------|---------------|
| MFU | 33.5% | ~45~47% |
| 처리속도 | 292K tok/s | ~390~410K tok/s |
| 50B 토큰 학습 | ~47시간 | ~34~36시간 |
| 핵심 원인 | NUMA misalignment (69% 크로스-NUMA) | NUMA affinity 적용으로 해결 |

---

## 6. 적용 우선순위

| 순위 | 항목 | 효과 | 난이도 | 비고 |
|------|------|------|--------|------|
| 1 | **NUMA affinity** | +4~9% | 낮음 | 코드 변경 최소, 효과 최대 |
| 2 | **QKV fusion** | +8~12% | 중간 | 모델 코드 수정 필요 |
| 3 | **NCCL 환경변수** | +1~2% | 낮음 | 한 줄 추가, 즉시 적용 |
| 4 | **DataLoader NUMA 정렬** | +1~2% | 낮음 | pin_memory_device 추가 |
| 5 | **FA2 native RoPE** | +3~5% | 중간 | FA2 버전 확인 후 적용 |

---

## 7. 참고

- Phase 3 ORPO 완료 후, 다음 프리트레인 런 전에 **NUMA affinity를 먼저 적용** 권장
- `launch_3b_pretrain.sh`와 `pretrain.py` 파일 수정 필요
- 실제 적용 시 `launch_3b_pretrain.sh`, `pretrain.py` 파일 내용을 기반으로 구체적 수정안 작성 예정
