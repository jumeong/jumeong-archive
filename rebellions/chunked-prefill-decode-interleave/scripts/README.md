# 재현 코드

- compile_model.py: 고정 revision의 원본 모델 다운로드 및 NPU 모델 컴파일.
- run_experiment.py: none/fixed 정책 측정. SOURCE·PYTHON 경로를 실행 환경에 맞게 수정해야 함.
- measure.py: SSE 수집과 지연 통계, 개별 trace 생성. aiohttp 필요.
- report.py: 저장된 원시 결과로 표와 비교 trace 생성. Python 표준 라이브러리만 사용.
- plot_results.py: 그래프 생성. matplotlib 필요.

모델·가상환경·로그는 포함하지 않습니다. 환경 준비와 기준 PR은 [루트 README](../README.md)를 참조하세요.
