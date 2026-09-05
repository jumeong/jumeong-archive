# Core Concepts

vLLM은 세 가지 핵심 병렬 처리 전략을 제공합니다: **텐서 병렬화(Tensor Parallelism, TP)**, **파이프라인 병렬화(Pipeline Parallelism, PP)**, **데이터 병렬화(Data Parallelism, DP)** [[1](https://rocm.blogs.amd.com/software-tools-optimization/vllm-moe-guide/README.html#references), [5](https://rocm.blogs.amd.com/software-tools-optimization/vllm-moe-guide/README.html#references)]. 

> **참고**: 전문가 병렬화(Expert Parallelism, EP)는 MoE(Mixture of Experts) 모델을 위한 특수 플래그로, TP 또는 DP와 결합하여 작동합니다.

---

## 1. Tensor Parallelism (TP)

- **원리 (What it does)**: 개별 레이어를 여러 GPU에 샤딩(분할)합니다. 각 GPU는 각 레이어의 일부를 처리하며, 집합 통신(collective communication)을 통해 결과를 동기화합니다.

![Figure 1: Data flow for Tensor Parallelism when Tensor parallel=4](https://rocm.blogs.amd.com/_images/tp_vanilla_improved.svg)  
*Figure 1: Data flow for Tensor Parallelism when Tensor parallel=4*

- **Figure 1 설명**:
  - 단일 요청 $X$의 텐서가 4개의 GPU에 나뉘어 분할됩니다.
  - 모든 GPU가 **동일한 계산**을 협력하여 수행하며, 각 GPU는 각 레이어 가중치의 1/4을 보유합니다.
  - 통신을 위해 각 레이어 이후 `AllReduce` 작업이 필요합니다.

- **사용 사례 (Use case)**:
  - 단일 모델이 너무 커서 GPU 1장에 들어가지 않을 때 사용합니다.
  - 단일 요청을 여러 GPU가 병렬로 처리하므로 **지연 시간(Latency)을 단축**할 수 있습니다.

- **제약 사항 (Constraint)**:
  - 어텐션 헤드(Attention heads) 수의 제약을 받으며, 병렬 크기로 나누어 떨어져야 합니다.
  - *예시*: TP=3을 사용할 경우, 다음과 같은 에러가 발생할 수 있습니다.
    > `Error message: Total number of attention heads (64) must be divisible by tensor parallel size (3)`

---

## 2. Data Parallelism (DP)

- **원리 (What it does)**: 모델의 완전한 복제본(Replica)을 여러 개 생성하여 각 GPU가 서로 다른 요청을 독립적으로 처리합니다. 여러 요청을 동시에 처리함으로써 처리량(Throughput)을 높입니다.

![Figure 2: Data flow for Data Parallelism for DP=4](https://rocm.blogs.amd.com/_images/data_parallelism.svg)  
*Figure 2: Data flow for Data Parallelism for DP=4*

- **Figure 2 설명**:
  - 여러 개의 서로 다른 요청이 동시에 처리됩니다.
  - 각 GPU는 완전하고 독립적인 복제본으로, 서로 다른 요청을 독립적으로 처리하여 동시 요청에 대해 **4배의 처리량**을 달성합니다.
  - GPU 간의 통신이 전혀 필요하지 않습니다.

- **사용 사례 (Use case)**:
  - 더 높은 처리량(Throughput)이 필요하고, 독립적으로 처리 가능한 요청들이 있을 때 사용합니다.

- **제약 사항 (Constraint)**:
  - 일반적으로 지연 시간(Latency)은 감소시키지 않습니다.

---

## 3. Pipeline Parallelism (PP)

- **원리 (What it does)**: 모델의 레이어를 여러 GPU 또는 노드에 나누어 배치하고, 각 GPU가 서로 다른 레이어를 순차적으로 처리합니다. 데이터는 조립 라인(Assembly line)처럼 이러한 단계를 거쳐 흐릅니다.

![Figure 3: Data flow for Vanilla Pipeline Parallelism for PP=4](https://rocm.blogs.amd.com/_images/vanilla_pipeline_parallelism.svg)  
*Figure 3: Data flow for Vanilla Pipeline Parallelism for PP=4*

- **Figure 3 설명 (Vanilla PP 기준)**:
  - 개념 설명을 위한 순수(Vanilla) 파이프라인 병렬화에서는 하나의 요청이 순차적으로 처리됩니다.
  - 한 번에 하나의 GPU만 활성화되고 다른 GPU는 유휴 상태(Idle, 파이프라인 버블)로 대기합니다.
  - 요청당 총 지연 시간은 4단계의 순차적 스텝이 걸리며, GPU 활용도는 25%(4대 중 1대만 활성)에 불과합니다.

### 특징 및 문제점

- **문제점 (The Problem)**:
  - 한 번에 하나의 요청만 순차 처리하는 Vanilla PP 환경에서는 GPU의 75%가 유휴 상태로 낭비됩니다.

- **vLLM의 최적화 (vLLM’s Optimization)**:
  - vLLM은 파이프라인을 통해 **여러 요청을 동시에 처리**합니다.
  - GPU 0이 요청 B를 처리하는 동안 GPU 1은 요청 A를 처리하고, GPU 2는 그보다 앞선 요청을 처리하는 방식을 취합니다.
  - 이를 통해 파이프라인의 모든 단계가 동시에 가동되도록 유지하여 유휴 시간을 획기적으로 줄입니다.

- **핵심 특성 (Key Characteristics)**:
  - 각 GPU는 모델 레이어의 일부("파이프라인 단계")를 보유합니다.
  - 데이터는 파이프라인 단계를 순차적으로 통과합니다.
  - **단일 요청 지연 시간**: TP보다 깁니다 (모든 단계를 순차적으로 거쳐야 함).
  - **처리량**: 파이프라인이 동시 요청들로 가득 찼을 때 높습니다.
  - **가장 적합한 환경**: 노드 내부에는 TP를 적용하고, 노드 간에는 PP를 결합하는 다중 노드(Multi-node) 배포 환경.

- **사용 시점 (When to use)**:
  - 모델이 너무 커서 단일 노드에 들어가지 않을 때 (노드 내 TP + 노드 간 PP 결합).
  - 2의 거듭제곱이 아닌 GPU 수(3, 5, 6, 7, 9 등)로 배포해야 할 때.

- **제약 사항 (Constraint)**:
  - 일반적으로 지연 시간(Latency)은 감소시키지 않습니다.

---

## 4. Expert Parallelism for MoE Models

### Why Expert Parallelism Reduces Latency for MoE Models
- **핵심 통찰 (The Key Insight)**: MoE 모델의 병목은 연산량(Compute)이 아니라 **메모리 대역폭(Memory Bandwidth)**입니다.
  - MoE 모델은 희소 활성화(Sparse Activation) 특성을 가집니다 (예: DeepSeek은 토큰당 671B 파라미터 중 37B만 사용).
  - GPU는 계산보다 메모리에서 전문가 가중치(Expert Weights)를 로드하는 데 대부분의 시간을 소비합니다.
  - 단일 GPU의 메모리 대역폭 한계로 인해 전문가에 접근할 수 있는 속도가 제한됩니다.
- **해결책 (Solution)**: 전문가들을 여러 GPU에 분산 배치하여 총 집합 메모리 대역폭(Aggregate Memory Bandwidth)을 확보합니다.
  - 8대의 GPU에 EP를 활성화하면 전문가 가중치 로딩에 대해 **8배의 메모리 대역폭**을 확보할 수 있습니다.
  - 이를 통해 MoE 모델의 주된 지연 요인인 메모리 전송 대기 시간을 줄입니다.
- **통신 트레이드오프 (Communication Tradeoff)**:
  - EP는 토큰을 서로 다른 GPU에 있는 전문가에게 라우팅하기 위해 `AllToAll` 통신이 필요합니다.
  - 그러나 대부분의 모델에서 이 통신 비용은 메모리 대역폭 절감 효과에 비해 훨씬 작습니다 (예외: 활성화 밀도가 1% 미만인 극도로 희소한 모델).

---

### TP + EP: Tensor Parallelism for MoE Models

아래 Figure 4와 같이 TP=8 및 Expert Parallelism을 활성화하면 전문가들이 8개의 GPU에 분할(GPU당 32명의 전문가)되며, 각 레이어 이후 `AllReduce` 통신이 사용됩니다.

![Figure 4: Data flow for TP=8 on a single node (with Expert Parallelism)](https://rocm.blogs.amd.com/_images/tp_moe_EP_v2.svg)  
*Figure 4: Data flow for TP=8 on a single node (with Expert Parallelism)*

- **사용 시점 (When to use TP+EP)**:
  - 단일 GPU에 들어가지 않는 대형 MoE 모델
  - 지연 시간(Latency)이 가장 중요한 저~중간 수준의 동시성(Concurrency) 워크로드
  - KV 캐시 복제(Duplication)를 처리할 수 있는 충분한 HBM이 확보된 경우

---

### DP + EP: DP Attention with Expert Parallelism

- 서로 다른 GPU가 서로 다른 요청을 처리합니다 (요청 수준 병렬화, Request-level Parallelism).
- **KV 캐시가 GPU 간에 분할(Partitioned)**됩니다: 각 GPU는 할당된 요청에 대한 캐시만 보유합니다.
- Non-MoE 레이어는 복제(Replicated)되지만 KV 캐시는 분할된 상태를 유지합니다.
- MoE 전문가는 모든 GPU에 걸쳐 분산 배치됩니다.
- 레이어 간 통신은 `AllToAll`을 통해 이루어집니다.

이 아키텍처는 아래 Figure 7에 나와 있으며, 전문가들이 여러 GPU에 분산되고 `AllToAll` 통신을 통해 분할된 KV 캐시로 효율적인 요청 수준 병렬 처리를 가능하게 합니다.

![Figure 7: Data flow for DP=8 with Expert Parallelism](https://rocm.blogs.amd.com/_images/dp_with_ep_moe_v2.svg)  
*Figure 7: Data flow for DP=8 with Expert Parallelism*

- **사용 시점 (When to use DP+EP)**:
  - KV 캐시 메모리가 중요한 MLA / MQA 모델에 필수적
  - TP 구성 옵션이 호환되지 않는 경우 (2의 거듭제곱이 아닌 경우. 예: Qwen3-Coder-480B FP8, MiniMax-M2)
  - Non-expert 레이어가 단일 GPU에 적재 가능한 대형 MoE 모델에 권장
  - 지연 시간보다 높은 QPS(초당 쿼리 수)가 더 중요한 처리량(Throughput) 중심의 배포 환경
