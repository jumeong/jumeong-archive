# Qwen3-4B BF16 교대 실험

먼저 [결과 요약](reports/SUMMARY.ko.md)을 읽고 [비교 그래프](reports/tradeoff.png)를 보세요.
핵심은 긴 prefill이 기존 요청의 출력을 오래 멈추게 하는 현상을 줄이는 것입니다.

## 기준 PR과 코드

이 실험은 [hello-npu/vllm-rbln PR #1 — feat(optimum): interleave prefill chunks with decode steps (opt-in)](https://github.com/hello-npu/vllm-rbln/pull/1)을 기반으로 진행했습니다.

- 브랜치: `feat/c1-prefill-decode-interleave`
- 실험에 사용한 커밋: [`a84bd4cf727459db306fd3f4cb4d191dcd29054a`](https://github.com/hello-npu/vllm-rbln/commit/a84bd4cf727459db306fd3f4cb4d191dcd29054a)
- 변경 내용: optimum 경로에서 prefill을 청크 단위로 처리하고 decode 스텝과 교대하는 선택 기능.
- 비교: 같은 PR 커밋에서 `VLLM_RBLN_INTERLEAVE_POLICY=none` 대 `VLLM_RBLN_INTERLEAVE_POLICY=fixed`, `VLLM_RBLN_INTERLEAVE_K=1`.
- 두 조건 모두 PR 소스를 `PYTHONPATH`로 선택했습니다. 설치된 배포판과 PR 코드의 비교가 아닙니다.
- 모델: 원본 [Qwen/Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B), BF16 export, 양자화 없음.
- Python 3.12.3, vLLM 0.13.0, optimum-rbln/rebel-compiler 0.10.1. 상세 환경과 모델 revision은 `metadata/`에 기록했습니다.

## 포함된 파일

| 폴더 | 용도 |
|---|---|
| reports/ | 읽을 결과물: 한국어 요약, 상세 통계 REPORT.md, comparison.csv, PNG/SVG 그래프 |
| perfetto/injection/ | 핵심 실험: 긴 프롬프트 주입 시 none/fixed 타임라인 비교 |
| data/results/ | 요청별 토큰 ID·수신 시각(raw.json), 통계(summary.json), 컴파일 혼입 검사(validation.json), 개별 정책 trace |
| data/workloads/ | 두 정책에 동일하게 사용한 입력 토큰·도착 시각·출력 길이 |
| scripts/ | 모델 컴파일, 실험 실행, 통계 집계, 그래프 생성 코드 |
| metadata/ | 패키지 버전, 컴파일 설정, 통합 서버 명령, 결과 검증 기록 |

## Perfetto 시작 파일

[3072토큰 주입 비교](perfetto/injection/injection-3072-seed0.json)를
https://ui.perfetto.dev 의 Open trace file로 여세요.
none/fixed의 배경 요청과 새 요청을 비교할 수 있습니다.
이 trace는 클라이언트의 토큰 수신 간격이며 NPU 커널 실행 기록은 아닙니다.

## 파일 이름 읽기

- injection-2048-seed1: 긴 입력 2048토큰, 두 번째 반복.
- none/fixed: 비교 정책. fixed는 K=1.
- warmup 기록은 공유 자료에서 삭제했습니다. 실제 실험에서는 동시 요청 1~4개와 prefill/decode 간섭을 먼저 워밍업했고, 본 측정 중 추가 컴파일이 없음을 검사했습니다.

## Pilot: 주입 시점 결정

Pilot은 워밍업을 마친 서버의 속도를 확인하고 본 실험의 주입 시점을 정하는 예비 측정입니다.
`none`에서 짧은 입력 128토큰과 긴 입력 2048토큰을 각각 64토큰 생성했습니다.
짧은 요청의 TTFT 약 73ms와 ITL 중앙값 약 48ms를 사용했습니다.

```text
주입 시점 = max(1초, 3 × TTFT + 8 × ITL 중앙값) = 1초
```

계수 3·8과 최소 1초는 배경 요청의 생성이 시작될 시간을 확보하기 위한 경험적 기준이며,
PR에서 지정하거나 최적화로 찾은 값은 아닙니다. 주입 후 두 배경 요청의 첫 토큰 수신과
마지막 토큰 수신 사이에 긴 요청이 전송됐는지도 사후 검사했습니다.
이는 클라이언트 관측 검증이며 주입 순간의 NPU 커널 상태를 직접 확인한 것은 아닙니다.
Pilot 원시 결과는 공유 자료에 포함하지 않습니다. 설정 근거는 `metadata/pilot-settings.json`에 남겼습니다.
재실행 코드는 재현을 위해 pilot을 다시 수행합니다.

## 결과 다시 생성

컴파일 모델(`models/`), 그래프용 가상환경(`runtime/`), 실행 로그(`logs/`),
제외된 초기 측정(`archive/`)은 Git 공유용 자료에서 제거했습니다.
원시 결과와 보고서는 남아 있으므로 NPU 없이 통계와 그래프를 다시 생성할 수 있습니다.

실험 루트에서 다음과 같이 실행하세요. matplotlib은 기존 그래프의 재생성에만 필요합니다.

```sh
python3 scripts/report.py
python3 -m venv .venv-analysis
.venv-analysis/bin/python -m pip install matplotlib
.venv-analysis/bin/python scripts/plot_results.py
```

NPU 실험 자체를 재실행하려면 CA22와 호환 SDK 환경, 위 PR 커밋의 소스가 필요합니다.
`scripts/run_experiment.py`의 `SOURCE`와 `PYTHON`을 실행 환경에 맞게 수정하고,
`mkdir -p models logs metadata data/results data/workloads`로 출력 폴더를 준비한 뒤
`scripts/compile_model.py`로 모델을 다시 컴파일해야 합니다.
컴파일 스크립트는 Hugging Face에서 고정 revision의 가중치를 내려받습니다.
기존 결과를 덮어쓰지 않으므로 측정 재실행 전 결과·설정 파일은 별도로 보존하세요.

메타데이터의 절대 경로는 원래 실험 당시 기록입니다. 삭제되거나 이동한 경로도 포함하며,
현재 checkout에서 해당 파일이 제공된다는 뜻은 아닙니다.

검증 완료: 본 측정 18회, 요청 27쌍, 동일 출력 토큰 9,792개, Perfetto 비교 파일 9개.
