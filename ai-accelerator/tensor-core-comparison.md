# NVIDIA Tensor Core vs Google TPU Systolic Array 비교

AI 가속기의 핵심인 **NVIDIA Tensor Core**와 **Google TPU의 Systolic Array** 동작 원리를 비교한다.

---

## 1. 아키텍처 구조

### 1.1 NVIDIA Tensor Core

Tensor Core는 GPU SM(Streaming Multiprocessor) 내부에 위치하며, **한 사이클에 소규모 행렬 연산(MMA)**을 수행한다.

```
+---------------------------------------+
|       Streaming Multiprocessor (SM)   |
|                                       |
|   [CUDA Core] [CUDA Core] ... ×128    |
|                                       |
|   [Tensor Core] [Tensor Core]         |
|   [Tensor Core] [Tensor Core]  ×4     |
|   D = A x B + C  (MMA per cycle)      |
|                                       |
|   [  Shared Memory / L1 Cache  ]      |
+---------------------------------------+
```

MMA의 타일 크기는 세대마다 다르다:

| 세대 | 아키텍처 | SM당 TC 수 | 단위 MMA 크기 | 1 TC의 FMA/cycle |
|------|---------|-----------|-------------|------------------|
| 1세대 | Volta (V100) | 8개 | 4×4×4 | 64 |
| 2세대 | Turing (T4) | 8개 | 4×4×4 | 64 |
| 3세대 | Ampere (A100) | 4개 | 8×4×8 | 256 |
| 4세대 | Hopper (H100) | 4개 | 8×4×16 | 512 |
| 5세대 | Blackwell (B200) | 4개 | - (더 확장) | 1024+ |

> 세대가 올라갈수록 단일 Tensor Core가 처리하는 타일이 커지고,
> 대신 SM당 TC 개수는 8 → 4개로 줄어드는 추세이다.

- Warp(32 threads) 단위로 협력하여 더 큰 행렬 타일을 처리
- SM당 여러 개의 Tensor Core가 병렬로 동작

### 1.2 Google TPU Systolic Array

TPU의 핵심은 **Systolic Array**로, PE(Processing Element)가 2D 격자 형태로 배열되어 있고, 데이터가 매 클럭마다 인접 PE로 전달되며 연산한다.

```
              Partial Sum 흐름 (위 → 아래)
                |       |       |       |
                v       v       v       v
          +-------+-------+-------+-------+
x[0] -->  | PE    | PE    | PE    | PE    |  --> (Activation 전달)
          | w[0,0]| w[0,1]| w[0,2]| w[0,3]|
          +-------+-------+-------+-------+
x[1] -->  | PE    | PE    | PE    | PE    |
          | w[1,0]| w[1,1]| w[1,2]| w[1,3]|
          +-------+-------+-------+-------+
x[2] -->  | PE    | PE    | PE    | PE    |
          | w[2,0]| w[2,1]| w[2,2]| w[2,3]|
          +-------+-------+-------+-------+
x[3] -->  | PE    | PE    | PE    | PE    |
          | w[3,0]| w[3,1]| w[3,2]| w[3,3]|
          +-------+-------+-------+-------+
                |       |       |       |
                v       v       v       v
              y[*,0]  y[*,1]  y[*,2]  y[*,3]  (출력)

Activation: 왼쪽 → 오른쪽 (각 행으로 입력)
Partial Sum:     위 → 아래 (각 열에서 누적)
Weight:     PE에 미리 로드 (고정)
```

- **TPU v1**: 256×256 Systolic Array (65,536 MAC units)
- **Weight Stationary** 방식: 가중치를 PE에 미리 로드 → Activation이 왼쪽에서 오른쪽으로 흐름
- 각 PE는 **MAC(Multiply-Accumulate)** 수행 후 Partial Sum을 아래 PE로 전달

---

## 2. 성능 비교

### 2.1 단일 유닛 연산 성능

| 항목 | Tensor Core 1개 (Volta) | Tensor Core 1개 (Hopper) | Systolic PE 1개 |
|------|------------------------|-------------------------|----------------|
| **1 cycle 연산량** | 64 FMA (128 FLOP) | 512 FMA (1,024 FLOP) | 1 MAC (2 OPs) |
| **내부 곱셈기 수** | 64개 (4×4×4) | 512개 (8×4×16) | 1개 |
| **Accumulator** | 16개 (4×4) | 32개 (8×4) | 1개 |
| **데이터 입력** | Warp가 레지스터에서 공급 | Warp가 레지스터에서 공급 | 인접 PE에서 자동 전달 |

### 2.2 칩 전체 연산 성능

```
◆ NVIDIA H100 (SXM, Hopper 4세대)
  SM 수: 132
  SM당 Tensor Core: 4개
  총 Tensor Core: 132 x 4 = 528개
  클럭: ~1.83 GHz (Boost)

  단일 TC 처리량 (FP16):
    1,024 FLOP x 1.83 GHz = 1,875 GFLOPS

  칩 전체:
    FP16: 528 x 1,875 GFLOPS ≈ 990 TFLOPS
    FP8:  ~1,979 TFLOPS
    TF32: ~495 TFLOPS
    TDP:  ~700W

◆ Google TPU v1
  Array 크기: 256 x 256 = 65,536 PE
  클럭: 700 MHz

  단일 PE 처리량 (INT8):
    2 OPs x 700 MHz = 1.4 GOPS

  칩 전체 (INT8):
    65,536 x 1.4 GOPS ≈ 92 TOPS

◆ Google TPU v4
  Array 크기: 128 x 128 x 4 = 65,536 PE (추정)
  클럭: ~1.05 GHz (추정)

  칩 전체:
    BF16: 65,536 x 2 x 1.05 GHz ≈ 275 TFLOPS
    INT8: ~275 TOPS
    TDP:  ~175W (칩당)
```

### 2.3 행렬 크기별 연산 시간 비교

정방 행렬 곱 `C[N×N] = A[N×N] × B[N×N]` 기준, 순수 연산 시간만 비교 (메모리 지연 제외).

**총 필요 연산량**: 2 × N³ FLOP (곱셈 N³ + 덧셈 N³)

#### 단일 유닛 비교 (Tensor Core 1개 vs TPU v1 전체 Array)

```
Tensor Core 1개 (Hopper):
  처리량 = 1,024 FLOP/cycle, 클럭 = 1.83 GHz
  cycle 수 = 2N³ / 1,024
  시간 = cycle 수 / 1.83 GHz

TPU v1 Systolic Array (256×256):
  정상 상태에서 65,536 MAC/cycle = 131,072 FLOP/cycle
  클럭 = 700 MHz
  총 cycle = N + K + N - 2  (파이프라인 fill/drain 포함)
  ※ Array보다 큰 행렬은 타일링 필요
```

| N | 총 FLOP | TC 1개 (Hopper) | TPU v1 Array (256×256) |
|---|---------|-----------------|----------------------|
| **64** | 524K | 280 cycles → **0.15 μs** | 190 cycles → **0.27 μs** |
| **128** | 4.2M | 2,048 cycles → **1.1 μs** | 382 cycles → **0.55 μs** |
| **256** | 33.6M | 16,384 cycles → **8.9 μs** | 766 cycles → **1.09 μs** |
| **512** | 268M | 131,072 cycles → **71.6 μs** | 타일링 4회 × ~1,534 cycles → **8.8 μs** |
| **1024** | 2.1G | 1,048,576 cycles → **573 μs** | 타일링 16회 × ~3,070 cycles → **70.1 μs** |
| **4096** | 137G | 67,108,864 cycles → **36.7 ms** | 타일링 1,024회 × ~12,286 cycles → **17.9 ms** |

> **TC 1개 vs Array 전체**는 공정한 비교가 아니다. 아래 칩 레벨 비교 참고.

#### 칩 전체 비교 (H100 vs TPU v1)

```
H100: 528개 TC × 1,024 FLOP/cycle × 1.83 GHz = 990 TFLOPS
TPU v1: 65,536 PE × 2 OPs/cycle × 700 MHz = 92 TOPS
```

| N | 총 FLOP | H100 칩 전체 (이론) | TPU v1 칩 전체 (이론) |
|---|---------|-------------------|---------------------|
| **64** | 524K | **0.5 ns** | **5.7 ns** |
| **128** | 4.2M | **4.2 ns** | **45.7 ns** |
| **256** | 33.6M | **33.9 ns** | **365 ns** |
| **512** | 268M | **271 ns** | **2.9 μs** |
| **1024** | 2.1G | **2.2 μs** | **23.3 μs** |
| **4096** | 137G | **138.8 μs** | **1.49 ms** |

> 위 수치는 연산 유닛의 이론적 Peak 처리량 기준이다.
> 실제로는 **메모리 대역폭**, **타일링 오버헤드**, **파이프라인 fill/drain** 등으로 인해
> Peak 대비 50~80% 수준의 실효 성능을 보인다.

### 2.4 핵심 차이 요약

```
작은 행렬 (N < 256):
  - TC: 단일 유닛으로도 빠르게 처리 가능
  - TPU: Array 활용률 저하 (PE 대부분 유휴)
         256×256 Array에 64×64 행렬 → PE 활용률 6.25%

큰 행렬 (N >= 256):
  - TC: 타일 분할 후 수백 개 TC에서 병렬 처리
  - TPU: Array에 딱 맞거나 타일링하여 처리, 파이프라인 효율 상승
         행렬이 클수록 fill/drain 오버헤드 비율 감소

초대형 행렬 (N >= 4096, LLM 학습 등):
  - H100: 단일 칩 절대 성능 우위 (~990 TFLOPS)
  - TPU: Pod 확장으로 대응 (v4 4096칩 Pod → 수백 PFLOPS)
```

