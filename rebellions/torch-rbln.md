# CUDA와 RBLN에서의 PyTorch eager mode와 `torch.compile`

`torch-rbln`은 PyTorch에 `rbln` 장치와 `torch.rbln` API를 추가하고, PyTorch의 연산자와 컴파일 API를 Rebellions NPU 실행 경로에 연결한다. 장치와 컴파일 백엔드는 별개다. Eager mode에서 장치별 구현은 텐서의 dispatch key로 선택된다. `torch.compile`에서 그래프를 처리할 컴파일러는 `backend=`로 선택되며, PyTorch의 기본값은 `"inductor"`다. 아래 비교는 RBLN SDK 0.12 계열 문서와 공개 `torch-rbln` 소스를 기준으로 한다. [RBLN PyTorch 개요](https://docs.rbln.ai/latest/software/rbln_pytorch/index.html) · [PyTorch `torch.compile` API](https://docs.pytorch.org/docs/stable/generated/torch.compile.html)

| 실행 방식 | CUDA | RBLN |
| --- | --- | --- |
| Eager | ATen dispatcher → PyTorch의 CUDA 구현 → GPU | ATen dispatcher → `PrivateUse1` → `torch-rbln`의 네이티브 구현 또는 연산자 단위 컴파일 → NPU. 조건에 따라 CPU fallback |
| `torch.compile` | Dynamo → 기본 백엔드 `inductor` → GPU용 커널 및 라이브러리 호출 | Dynamo → 명시적으로 선택한 `backend="rbln"` → RBLN Compiler 및 런타임 → NPU. `torch-rbln`이 백엔드 등록과 호출 래퍼를 제공 |

## 1. PyTorch가 `torch-rbln`을 불러오는 시점

패키지의 `torch.backends` entry point를 통해 PyTorch가 확장을 자동으로 불러온다. 초기화 함수 `torch_backends_entry_point()`는 PyTorch가 외부 가속기용으로 제공하는 `PrivateUse1` 장치 이름을 `rbln`으로 설정하고, `torch.rbln` 모듈과 C++·Python 연산자 구현을 등록한다. 이때 `torch.compile` 래퍼도 설치한다. 장치 등록은 eager dispatch에서, 컴파일 래퍼는 `torch.compile` 호출에서 각각 사용된다. [패키지 entry point](https://github.com/RBLN-SW/torch-rbln/blob/main/pyproject.toml) · [초기화 코드](https://github.com/RBLN-SW/torch-rbln/blob/main/torch_rbln/__init__.py#L24-L107) · [PyTorch 확장 자동 로딩](https://docs.pytorch.org/tutorials/unstable/python_extension_autoload.html)

## 2. Eager mode

**CUDA.** `torch.relu(x + y)`를 호출하면 Python이 덧셈과 ReLU를 차례로 호출한다. CUDA 텐서의 경우 ATen dispatcher는 각 연산의 CUDA 구현 경로를 선택한다. 해당 구현은 연산에 맞는 GPU 커널을 실행하거나 cuBLAS 같은 라이브러리를 호출한다. [PyTorch dispatcher](https://docs.pytorch.org/tutorials/advanced/dispatcher.html) · [PyTorch CUDA 환경 변수와 라이브러리](https://docs.pytorch.org/docs/stable/cuda_environment_variables.html)

**RBLN.** 같은 호출에 RBLN 텐서를 넘기면 dispatcher가 `PrivateUse1`을 통해 `torch-rbln`에 등록된 구현으로 보낸다. 이후 경로는 연산자와 입력에 따라 갈린다. [RBLN eager dispatch](https://docs.rbln.ai/latest/software/rbln_pytorch/execution_semantics.html#operator-dispatch-path)

- **컴파일하는 연산**은 캐시 미스 시 연산 하나를 감싼 모듈을 내부적으로 `torch.compile(backend="rbln")`에 넘긴다. RBLN Compiler가 현재 입력에 맞는 실행 프로그램을 만들고 런타임이 NPU에서 실행한다. 같은 shape, dtype, stride 등의 입력 조건에서 프로그램을 재사용한다. C++ 캐시에 적중하면 Python과 Dynamo를 거치지 않고 런타임에 제출한다.
- **네이티브 연산**은 등록된 C++ 구현을 사용한다. 예를 들어 `view`는 텐서 메타데이터를 바꾸고, 지원되는 복사는 런타임을 통해 처리한다.
- **CPU fallback**은 eager NPU 경로가 지원하지 않는 연산이나 입력 조건에 적용될 수 있다. 필요한 데이터를 호스트로 가져와 해당 연산을 CPU에서 처리한다. 따라서 텐서의 장치가 `rbln`이어도 개별 연산이 CPU에서 실행될 수 있다.

사용자 코드가 eager인 상태에서 확장 내부의 컴파일 단위는 개별 연산이다. 덧셈과 ReLU가 각각 eager로 호출되면 두 연산을 함께 최적화할 그래프는 백엔드에 전달되지 않는다. [RBLN 실행 방식](https://docs.rbln.ai/latest/software/rbln_pytorch/execution_semantics.html#eager-and-graph-execution)

## 3. `torch.compile`

**CUDA와 `inductor`.** 첫 호출에서 TorchDynamo가 함수의 PyTorch 연산을 FX 그래프로 캡처한다. 기본 경로에서는 AOTAutograd 처리와 TorchInductor 최적화를 거쳐, Inductor가 연산을 융합하고 Triton 커널이나 기존 라이브러리 호출 등을 포함한 GPU 실행 코드를 만든다. [PyTorch 컴파일러 개요](https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/torch.compiler.html)

**RBLN과 `backend="rbln"`.** 래퍼는 필요하면 `rebel.core.torch_compile`을 불러와 `rbln` 백엔드를 등록한다. 원래 `torch.compile`이 Dynamo로 캡처한 그래프를 이 백엔드에 전달하면, `rebel-compiler`가 해당 그래프를 입력 shape에 맞춰 컴파일하고 RBLN 런타임이 실행한다. `torch-rbln`은 RBLN 텐서와 런타임을 연결하고, 컴파일된 callable을 감싸 오류 처리 등을 맡는다. 예를 들어 기본 정책에서는 컴파일 실패 시 원래 함수에 CPU로 옮긴 호출 인자를 넣어 실행할 수 있다. `TORCH_RBLN_DISABLE_FALLBACK=compile_error`를 설정하면 이 fallback 대신 오류가 드러난다. [RBLN 래퍼 코드](https://github.com/RBLN-SW/torch-rbln/blob/main/torch_rbln/_internal/monkey_patches.py#L28-L115) · [fallback 구현](https://github.com/RBLN-SW/torch-rbln/blob/main/torch_rbln/_internal/torch_compile_patch_helpers.py#L400-L421) · [RBLN 그래프 실행](https://docs.rbln.ai/latest/software/rbln_pytorch/execution_semantics.html#eager-and-graph-execution)

그래프 안의 덧셈과 ReLU는 하나의 그래프로 캡처되면 함께 컴파일된다. Graph break가 있으면 Dynamo가 캡처한 구간별로 백엔드에 넘기며, 그래프 밖의 연산이 RBLN 텐서에 대해 eager로 실행되면 다시 `PrivateUse1`을 거친다. RBLN 컴파일러가 그래프 안의 일부 연산을 호스트 연산으로 포함하는 경우도 있다. 이는 컴파일에 실패해 원래 함수를 CPU에서 다시 실행하는 fallback과 별개의 동작이다. [PyTorch graph break](https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/compile/programming_model.graph_breaks_index.html) · [RBLN 그래프 경계](https://docs.rbln.ai/latest/software/rbln_pytorch/execution_semantics.html#the-eager-and-graph-boundary)

## 4. 같은 함수에 네 가지 실행 방식 적용하기

아래 예시는 장치별 환경에서 `run("cuda")` 또는 `run("rbln")`을 호출한다. CUDA 쪽은 CUDA 지원 PyTorch를 사용한다. RBLN 쪽은 공식 설치 가이드가 지정한 CPU 빌드 PyTorch에 `torch-rbln`과 호환되는 `rebel-compiler` 및 런타임 환경을 사용한다. `device`는 텐서를 만들 때, `backend`는 함수를 컴파일할 때 지정한다. 모듈을 쓰는 경우에는 입력과 함께 파라미터 및 버퍼도 해당 장치로 옮긴다. [RBLN 설치 가이드](https://docs.rbln.ai/latest/software/rbln_pytorch/installation.html)

```python
import torch


def compute(x, y):
    return torch.relu(x + y)


def run(device):
    x = torch.randn((128, 128), device=device, dtype=torch.float16)
    y = torch.randn_like(x)

    if device == "rbln":
        compiled_compute = torch.compile(compute, backend="rbln", dynamic=False)
    else:
        compiled_compute = torch.compile(compute, backend="inductor")

    with torch.inference_mode():
        eager_result = compute(x, y)
        compiled_result = compiled_compute(x, y)

    return eager_result, compiled_result
```
