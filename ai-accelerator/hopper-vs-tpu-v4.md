## 1. NVIDIA Hopper 아키텍처 (H100 SXM5 기준)

*   **단일 Tensor Core 연산 규격 (M, N, K):** **16 x 8 x 32**
*   **Cycle당 처리량:** **1,024 MACs / Cycle** (단일 Tensor Core 기준, INT8)
*   **칩 내부 구조:**
    *   칩 내 SM 개수: **132개**
    *   1개 SM 내 Tensor Core 개수: **4개**
    *   **칩 내 전체 Tensor Core 개수: 528개** (132 x 4)

*   **전체 연산 성능 (TOPS) 계산:**
    *   **Peak MACs / Cycle:** 540,672 MACs (528개 × 1,024)
    *   **Peak OPs / Cycle:** 1,081,344 OPs (1 MAC = 2 OPs)
    *   **Clock Frequency:** 약 1.83 GHz (Boost Clock 기준)
    *   **최대 처리량 (INT8 Dense 기준):** 1,081,344 OPs × 1.83 GHz ≈ **1,979 TOPS**

> **Reference:**
> *   https://resources.nvidia.com/en-us-hopper-architecture/nvidia-h100-tensor-c
> *   https://docs.nvidia.com/cuda/parallel-thread-execution/index.html

---

## 2. Google TPU v4 아키텍처

*   **단일 MXU 물리적 규격:** **128 x 128** (고정형 2차원 Systolic Array)
*   **Cycle당 처리량:** **16,384 MACs / Cycle** (단일 MXU 기준, INT8/BF16)
*   **칩 내부 구조:**
    *   칩 내 TensorCore 개수: **2개** (※ 구글은 칩 내부의 메인 프로세서 블록을 'TensorCore'라는 고유명사로 부름. NVIDIA의 Tensor Core보다 훨씬 큰 단위이며, NVIDIA 구조와 굳이 비교하자면 큰 SM에 대응시킬 수도 있음)
    *   1개 TensorCore 내 MXU 개수: **4개**
    *   **단일 칩 전체 MXU 개수: 8개** (2 x 4)

*   **전체 연산 성능 (TOPS) 계산:**
    *   **Peak MACs / Cycle:** 131,072 MACs (8개 × 16,384)
    *   **Peak OPs / Cycle:** 262,144 OPs (1 MAC = 2 OPs)
    *   **Clock Frequency:** 약 1.05 GHz
    *   **최대 처리량 (INT8 기준):** 262,144 OPs × 1.05 GHz ≈ **275 TOPS**

> **Reference:**
> *   https://arxiv.org/abs/2304.01433
> *   https://jax-ml.github.io/scaling-book/tpus/

---

## 3. Hopper vs TPU v4 시나리오 비교
두 하드웨어 간의 세대 차이(체급)를 배제하고 **순수 아키텍처의 구조적 효율성**을 비교하기 위해, 두 하드웨어의 전체 병렬 처리량이 완벽하게 동일해지도록 Hopper의 SM 개수를 32개로 제한하여 가정한다.
(Hopper 32개 SM: 131,072 MACs/Cycle vs TPU 8개 MXU: 131,072 MACs/Cycle)

### 3.1. Hopper가 유리한 시나리오: 불규칙하고 작은 GEMM 다수 발생 (예: MoE 추론)

MoE(Mixture of Experts) 추론에서 Router가 토큰을 8개의 Expert로 분산한 결과, 다음과 같은 작은 GEMM들이 칩 내부에서 동시다발적으로 수행된다고 가정한다.
각 Expert는 서로 다른 Weight를 사용하므로 8개 모두 완전히 독립적인 연산이다.

> Expert 0 : 96 × 128 × 128 GEMM  
> Expert 1 : 37 × 128 × 128 GEMM  
> Expert 2 : 142 × 128 × 128 GEMM  
> Expert 3 : 11 × 128 × 128 GEMM  
> Expert 4 : 205 × 128 × 128 GEMM  
> Expert 5 : 64 × 128 × 128 GEMM  
> Expert 6 : 89 × 128 × 128 GEMM  
> Expert 7 : 12 × 128 × 128 GEMM

위 8개 Expert의 유효 토큰 수(M) 총합은 **656개**이며, 이를 실제 연산량으로 환산하면 약 1,074만 MACs가 된다. 이 불규칙한 데이터가 동시에 투입되었을 때 발생하는 지연 시간을 수치화하면 다음과 같다.

| 항목 | TPU v4 풀칩 (8개 MXU) | Hopper (32개 SM 제한) |
|---|---|---|
| **실제 유효 연산량** | 약 1,074만 MACs (M 총합 656) | 약 1,074만 MACs (M 총합 656) |
| **이상적인 순수 계산** | 약 82 Cycles (1,074만 ÷ 13.1만) | 약 82 Cycles (1,074만 ÷ 13.1만) |
| **TPU (병렬) 소요 Cycle** | **약 459 Cycles**<br>(가장 긴 M=205 동기화 + Fill/Drain 254) | 해당 없음 |
| **TPU (직렬) 소요 Cycle** | **약 2,688 Cycles (최악의 경우)**<br>(M=656 순수 연산 + Fill/Drain 8회 누적) | 해당 없음 |
| **Hopper 소요 Cycle** | 해당 없음 | **약 82 Cycles + α (수학적 하한)**<br>(실제로는 Scheduling/Memory Overhead 일부 추가됨) |

* **TPU에게 불리한 점 (정적 스케줄링 제약과 Array Utilization 저하):** TPU v4 칩 내부의 8개 MXU는 완전히 독립적이지 않으며, 4개씩 코어(TensorCore)에 묶여 단일 프로그램(Single Program) 및 정적 스케줄링(Static Scheduling)을 공유한다. 이로 인해 8개의 불규칙한 Expert 연산을 처리할 때 구조적인 비효율이 발생한다.
    * **병렬 처리 시 (Runtime 불확실성으로 인한 낭비):** 8개의 MXU에서 동시에 연산을 돌리면 Fill/Drain Overhead(254 Cycle)는 1번만 감당하면 된다. 하지만 MoE 환경에서는 Expert별 토큰 수가 '런타임'에 동적으로 결정되므로, TPU의 강점인 '컴파일 시점의 정적 분배 및 최적화(Packing/Fusion)'를 완벽하게 적용하기가 매우 어렵다. 결국 가장 처리가 오래 걸리는 작업(M=205)이 끝날 때까지 다른 MXU들은 빈 사이클을 돌며 대기해야 하므로, 불균형한 토큰 분포에서는 필연적으로 심각한 Array Utilization(활용률) 저하가 발생한다.
    * **직렬 처리 시 (Pipeline Overhead 누적):** 위와 같은 패딩 낭비를 피하기 위해 어쩔 수 없이 각 Expert를 완전히 독립적으로 순차 실행하는 **최악의 경우(Worst-case)**, 이번에는 매 작업마다 거대한 Array를 비우고 채우는 Fill/Drain Overhead(254 Cycle)가 8번 연속으로 누적되어 전체 소요 Cycle이 2,688 수준으로 폭증하게 된다.
* **Hopper에게 유리한 점 (Warp Scheduler 유연성 및 Latency 은닉):** Hopper의 SM은 각각 독자적인 Warp Scheduler를 가진 완전히 독립적인 유닛이다. 제한된 32개의 SM만 사용하더라도, 8개의 Expert 연산을 자유롭게 쪼개어 높은 병렬성을 유지할 수 있다.
    * M=11짜리 작업이 끝나면 해당 SM들은 즉시 다음 작업을 수행하거나 다른 곳에 투입될 수 있으며, 가장 긴 작업(M=205)의 속도에 억지로 맞출 필요가 전혀 없다.
    * Tensor Core는 TPU보다 훨씬 작은 Tile 단위(16x8x32)로 연산을 분할하므로 **Padding으로 인한 낭비가 상대적으로 매우 작다**. 또한 Pipeline Latency가 존재하지만 매우 짧으며, **Warp Scheduler의 Context Switching을 통해 대부분 은닉(Hidden)된다**.
    * **결론:** 두 칩의 물리적 처리 능력(Peak MACs/Cycle)이 131,072로 완벽히 동일하게 주어지더라도, TPU는 정적 스케줄링 제약으로 인해 실제 소요 시간이 459 Cycle 이상으로 늘어지는 반면, Hopper는 **이론적 하한선인 82 Cycle에 가까운 성능(82 + α)**을 대부분 방어해낸다. 이것이 MoE 추론 환경에서 Hopper의 아키텍처가 더 유리한 이유이다.
