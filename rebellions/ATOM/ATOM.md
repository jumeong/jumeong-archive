# ATOM 추론 가속기의 구조와 실행 방식

ATOM은 호스트가 PCIe로 보낸 명령을 받아 신경망을 실행하는 SoC다. 칩 전체에서는 **Command Processor(CP)**가 작업을 배분하고, 각 **Neural Engine**에서는 **Task Manager(TM)**가 선행 작업의 완료 여부를 확인한다. 두 구성 요소 모두 칩에 구현된 하드웨어 블록이다. CP의 동작에는 장치 펌웨어가 관여한다. [ATOM 백서 그림 1·2][white6] · [Rebellions의 펌웨어 설명][firmware]

## ATOM+와 ATOM-Max 카드 사양

아래 수치는 PCIe 카드 **한 장** 기준이다. ATOM+에는 NPU 칩 하나, ATOM-Max에는 네 개가 들어간다. 2024년 아키텍처 백서의 카드 사양표는 RBLN-CA12를 다루므로, ATOM+의 전력과 폼팩터는 해당 SKU인 RBLN-CA22 제품 사양을 따랐다. [SDK의 카드별 칩 수][rsd] · [ATOM+ 사양][plus] · [ATOM-Max 사양][max] · [2024년 백서 표 1][oldcard]

| 항목 | ATOM+ (RBLN-CA22) | ATOM-Max (RBLN-CA25) |
| --- | ---: | ---: |
| NPU 칩 수 | 1 | 4 |
| 최대 FP16 연산량 | 32 TFLOPS | 128 TFLOPS |
| 최대 INT8 / INT4 연산량 | 128 / 256 TOPS | 512 / 1,024 TOPS |
| GDDR6 용량 | 16 GB | 64 GB |
| GDDR6 대역폭 | 256 GB/s | 1,024 GB/s (칩 합산) |
| 호스트 인터페이스 | PCIe Gen5 x16, 64 GB/s | PCIe Gen5 x16, 64 GB/s |
| 최대 소비전력 | 85 W | 350 W |
| 크기·냉각 | FHFL 싱글 슬롯, 패시브 공랭 | FHFL 듀얼 슬롯, 패시브 공랭 |

Max에서 4배가 되는 사양은 카드의 최대 연산량과 메모리 용량·대역폭이다. 호스트와 카드 사이의 링크는 두 제품이 같다. Max 백서는 전용 케이블로 최대 8장의 카드를 직접 연결하는 구성도 설명한다. [ATOM-Max 백서 3–5쪽][maxpaper]

## Figure 1. ATOM SoC Architecture

![Figure 1. ATOM SoC 전체와 Neural Engine Cluster의 내부 구성][img1]

삼성 5nm로 제조한 ATOM 한 칩에는 **클러스터 2개 × 엔진 4개**, 총 8개의 Neural Engine이 있다. 각 클러스터는 L1 캐시와 엔진 간 연결망을 공유한다. 칩 전체에는 두 클러스터 외에 Task DMA, L2 공유 SRAM, CP, PCIe 인터페이스, GDDR6 컨트롤러, 보안 부팅을 담당하는 Root-of-Trust가 있다. 그림의 DRAM은 컨트롤러 밖에 표시된 외부 GDDR6다. 호스트 CPU에서 동작하는 드라이버가 명령 스트림을 장치에 전달하면, 장치 안의 CP가 이를 받아 연산과 데이터 전송 작업을 배분한다. [백서 그림 1, 6쪽][white6] · [ISSCC 논문 그림 2.4.1][issccfig] · [드라이버와 펌웨어][firmware]

## Figure 2. Neural Engine 구성

![Figure 2. 엔진 내부의 Task Manager, Neural DMA, Scratch Pad, Compute Units][img2]

| 엔진 내부 블록 | 역할 |
| --- | --- |
| Task Manager | 연산·DMA·동기화 작업을 큐에 보관하고 우선순위와 의존성을 확인한다. |
| Neural DMA | 엔진에서 쓸 프로그램, 가중치, 입력 등의 데이터를 스크래치패드로 옮기거나 결과를 내보낸다. |
| L0 Scratch Pad | 해당 엔진이 사용할 프로그램과 중간 데이터를 담는 4 MB 로컬 SRAM이다. |
| Compute Units | Tensor Processor가 텐서 연산을, Vector Processor가 벡터 연산을 수행한다. Load/Store Processor는 데이터 읽기·쓰기를 담당하며 Register Banks는 연산에 쓸 값을 보관한다. |

ISSCC 논문의 확대도는 엔진 하나의 Tensor Processor를 FP16 4 TFLOPS, INT8 16 TOPS로 표시한다. 엔진 내부에는 여러 프로그래머블 처리 요소가 분산되어 있고, 백서는 SIMD와 MIMD를 함께 사용한다고 설명한다. [백서 그림 2, 7쪽][white7] · [ISSCC 그림 2.4.1][issccfig] · [SDK의 Neural DMA 설명][profiler]

## Figure 3. 계층형 메모리 서브시스템

![Figure 3. Neural Engine의 로컬 메모리에서 외부 DRAM까지의 계층][img3]

- **L0**는 엔진 전용 스크래치패드다. 엔진당 4 MB, 8개 합계 32 MB이며, 백서가 제시한 최대 8 TB/s는 로컬 SRAM 접근 대역폭이다.
- **L1**은 가까운 엔진들이 공유하는 클러스터 단위 Neural Cache다. ISSCC 논문은 같은 클러스터 안에서 엔진끼리 직접 데이터를 전달하는 경로도 설명한다.
- **L2**는 칩 전체가 공유하는 32 MB SRAM이다. 여러 뱅크에 접근을 분산해 병렬로 읽고 쓸 수 있다. ISSCC에서 집계한 온칩 메모리 64 MB는 L0 합계 32 MB와 L2 32 MB를 더한 값이다.
- **외부 GDDR6**는 한 칩 기준 16 GB, 256 GB/s다. 예를 들어 한 엔진이 중간 결과를 L0에서 재사용하면 같은 값을 GDDR6에서 다시 읽는 비용을 줄인다.

출처는 백서 7쪽의 메모리 계층과 ISSCC의 데이터 경로 d0–d4다. [백서 그림 3][white7] · [ISSCC 그림 2.4.2·2.4.3][issccfig]

## Figure 4. 버스와 동기화 범위

![Figure 4. 엔진 내부, 엔진 간, DMA 및 CP 사이의 동기화 계층][img4]

**데이터 경로**는 텐서와 프로그램을 옮기고, **동기화 경로**는 의존하는 작업이 언제 진행할 수 있는지 알린다. 예를 들어 입력 타일을 옮기는 일은 DMA가 수행하고, 그 타일을 읽을 수 있는 시점을 맞추는 데에는 동기화 신호가 쓰인다. [백서 그림 1][white6] · [그림 4][white8] · [ISSCC 그림 2.4.2][issccfig]

| 경로 | 연결 범위와 역할 |
| --- | --- |
| Engine-to-Engine Cluster NoC | 같은 클러스터의 엔진 간 데이터 경로. 이웃 엔진이 생산한 값을 전달하거나 여러 엔진의 부분 결과를 합치는 데 쓴다. |
| System NoC | 칩 전체의 데이터 연결망. 클러스터, Task DMA, L2 공유 메모리, 외부 메모리 컨트롤러 등 사이의 전송을 지원한다. |
| L0 Intra-Core Sync (s0) | 한 엔진 내부의 명령 의존성을 처리한다. 프로그래머블 Load/Store Processor와 다른 처리 요소가 동기화 명령으로 실행 순서를 맞춘다. |
| L1 Sync Bus (s1) | 같은 클러스터의 서로 다른 엔진이 TM을 통해 명령 수준에서 동기화한다. |
| L2 Sync Bus (s2) | 엔진 클러스터와 Task DMA 사이의 의존성을 처리한다. 옅은 회색 선이 그림 4에서 TDMA까지 이어진다. |
| L3 Sync (s3) | 칩 내부의 작업 수준 동기화. 그림 4에서 CP와 Task DMA 쪽을 연결한다. |
| L4 Sync (s4) | 호스트 I/O와 관련된 명령의 동기화. 그림 4에서 CP와 Host DMA 쪽을 연결한다. |
| PCIe Gen5 x16 | 칩 바깥의 호스트 및 지원되는 카드 간 통신에 쓰는 인터페이스다. |

그림 4의 **TDMA**는 Task DMA, **HDMA**는 Host DMA다. TDMA는 장치 메모리 계층 사이의 작업 단위 전송을, HDMA는 호스트 메모리와 장치 사이의 전송을 담당한다. ISSCC 저자들은 엔진 간 전용 동기화 경로에서 최장 경로 지연을 16 ns로 보고했다. [백서 그림 4, 8쪽][white8] · [ISSCC §2.4 본문][isscc] · [SDK의 DMA 명령 설명][profiler]

## Command Processor와 Task Manager

**Command Processor는 칩 전체를 제어하는 하드웨어 프로세서다.** 호스트에서 받은 명령을 바탕으로 작업을 만들고, 자신이 처리해야 하는 의존성을 확인한 뒤 연산 엔진과 DMA에 작업을 배분한다. 예를 들어 호스트 데이터 도착을 기다리는 동안에도 그 데이터와 독립적인 명령은 먼저 발행할 수 있다. CP의 실행을 제어하는 펌웨어는 별도로 업데이트할 수 있다. [ISSCC §2.4 및 그림 2.4.4][isscc] · [Rebellions 펌웨어 설명][firmware] · [CP 펌웨어 업데이트 문서][flash]

**Task Manager는 각 Neural Engine 안에 있는 하드웨어 블록이다.** ISSCC의 실제 칩 블록도에는 작업 큐, 우선순위 제어, 의존성 제어가 표시된다. 백서 그림 5-2에서는 TM들이 엔진 가까이에서 의존성을 해결하면서, 한 엔진이 현재 타일을 계산하는 동안 다른 DMA 작업이 진행된다. 이렇게 하면 각 단계마다 CP까지 완료를 보고하고 다시 지시를 받는 왕복을 줄일 수 있다. [ISSCC 그림 2.4.1][issccfig] · [백서 그림 5-1·5-2, 9쪽][white9]

공개특허 US20250251973A1은 이를 구현할 수 있는 예를 더 자세히 설명한다. TM의 회로는 대기 큐에서 작업 기술자(task descriptor)를 꺼내 의존성 행렬을 조회한다. 선행 작업이 완료되면 완료 신호로 행렬을 갱신하고, 조건을 만족한 작업을 후속 큐에 넘긴다. 특허 그림 8의 실시예에서는 L0 메모리와 연산·로드/스토어 블록 사이에 전용 동기화 경로를 둔다. 이 큐와 행렬의 구체적인 설계는 특허의 실시예이며, 양산 ATOM에서 직접 확인되는 범위는 위 ISSCC 블록도까지다. [특허 원문 PDF][patpdf] · [동일 공개번호의 명세서·청구항 1][pat]

## ATOM 칩 안에 Arm이나 RISC-V CPU가 있는가?

**Arm 탑재를 뒷받침하는 구체적인 자료는 ATOM의 공동 설계사 SEMIFIVE의 발표다.** SEMIFIVE는 ATOM을 자사 5nm HPC SoC 플랫폼을 사용해 상용화한 제품으로 소개하고, 그 플랫폼에 **64비트 Arm Cortex-A53 쿼드코어 CPU 클러스터**가 통합돼 있다고 명시한다. 따라서 ATOM이 Arm CPU를 포함한 플랫폼을 기반으로 한다는 근거가 있다. 다만 발표에서 A53 네 코어는 플랫폼 사양으로 제시된다. ATOM의 각 SKU에서 활성화된 CPU 코어 수와 **CP가 그 A53에서 구현됐는지**는 확인한 공개 자료만으로 특정하기 어렵다. [SEMIFIVE의 ATOM 상용화 발표][semifive] · [ATOM SoC 블록도][white6]

**RISC-V 탑재 여부는 미확인이다.** ISSCC 논문은 Neural Engine 내부에 분산된 *RISC-type* 처리 코어를 기술하고, Rebellions는 이 엔진용 프로그램이 자체 **Rebellions RISC ISA**를 대상으로 생성된다고 설명한다. 두 자료에서 확인되는 것은 엔진의 명령어 설계 방식이다. 어느 쪽도 ATOM에 RISC-V ISA를 구현한 CPU가 있다고 명시하지 않는다. 호스트에서 PyTorch와 드라이버를 실행하는 서버 CPU는 Figure 1의 PCIe 바깥에 있다. [ISSCC §2.4][isscc] · [Rebellions 컴파일러 설명][compiler] · [백서 그림 1][white6]

ISSCC 논문은 저장소에 포함된 PDF의 12–13쪽에서 볼 수 있다. [ISSCC Digest 원본](https://iccircle.com/static/upload/img20240529101639.pdf#page=12)

[white6]: https://rebellions.ai/wp-content/uploads/2024/07/ATOMgenAI_white-paper.pdf#page=6
[white7]: https://rebellions.ai/wp-content/uploads/2024/07/ATOMgenAI_white-paper.pdf#page=7
[white8]: https://rebellions.ai/wp-content/uploads/2024/07/ATOMgenAI_white-paper.pdf#page=8
[white9]: https://rebellions.ai/wp-content/uploads/2024/07/ATOMgenAI_white-paper.pdf#page=9
[oldcard]: https://rebellions.ai/wp-content/uploads/2024/07/ATOMgenAI_white-paper.pdf#page=5
[img1]: https://rebellions.ai/wp-content/uploads/2024/07/Kr_jp_en_atom_genai_architecture_sweet_spot-3-1024x411.png
[img2]: https://rebellions.ai/wp-content/uploads/2024/07/Kr_jp_en_atom_genai_architecture_sweet_spot-4.png
[img3]: https://rebellions.ai/wp-content/uploads/2024/07/Kr_jp_en_atom_genai_architecture_sweet_spot-5-1024x355.png
[img4]: https://rebellions.ai/wp-content/uploads/2024/07/Kr_jp_en_atom_genai_architecture_sweet_spot-6.png
[plus]: https://rebellions.ai/rebellions-product/rbln-ca22/
[max]: https://rebellions.ai/rebellions-product/atom-max/
[maxpaper]: https://rebellions.ai/wp-content/uploads/2025/04/White-Paper_Issue5_ATOM-Max_v1.0.pdf#page=3
[rsd]: https://docs.rbln.ai/latest/software/optimum/index.html
[isscc]: ./isscc2024.pdf#page=12
[issccfig]: ./isscc2024.pdf#page=13
[profiler]: https://docs.rbln.ai/v0.12.0/software/profiler/architecture.html
[patpdf]: https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/20250251973
[pat]: https://patents.google.com/patent/US20250251973A1/en
[firmware]: https://rebellions.ai/developers/
[flash]: https://docs.rbln.ai/v0.12.0/software/system_management/management_tools/rbln_flash.html
[semifive]: https://semifive.com/company/newsroom/press-releases/semifive-announces-commercialization-of-its-5nm-hpc-soc-platform-with-lead-partner-rebellions-ai-chipmaker-startup-based-in-korea/
[compiler]: https://rebellions.ai/understanding-rbln-compiler/
