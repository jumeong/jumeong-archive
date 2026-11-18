# vllm-rbln 플러그인의 역할

`vllm-rbln`은 vLLM이 Rebellions의 RBLN NPU에서 모델을 실행할 수 있게 하는 플랫폼 플러그인이다. vLLM은 서버에서 요청을 받고, 스케줄러로 어떤 요청을 처리할지 정하고, executor를 통해 워커에 모델 실행을 요청한다. `vllm-rbln`은 이 흐름에 `RblnPlatform`과 RBLN 워커를 제공한다. 워커가 모델 실행을 마치면 결과는 vLLM 엔진으로 돌아간다.

플랫폼 선택은 초기화 과정에서 한 번 일어난다. 이후 vLLM의 공통 코드가 플랫폼 객체의 메서드를 호출하거나, 설정에 지정된 워커를 실행할 때 RBLN 구현이 동작한다.

## 1. 패키지를 설치하면 플러그인이 등록된다

`vllm-rbln`의 `pyproject.toml`에는 다음 설정이 있다.

```toml
[project.entry-points."vllm.platform_plugins"]
rbln = "vllm_rbln:register"
```

설치 시 이 정보가 Python 패키지 메타데이터에 기록된다. `vllm.platform_plugins`는 vLLM이 조회하는 그룹 이름이고, `rbln`은 그 그룹 안의 플러그인 이름이다. `vllm_rbln:register`는 `vllm_rbln` 모듈의 `register` 함수를 가리킨다.

실제 `register()` 함수는 플랫폼 클래스의 경로를 문자열로 반환한다.

```python
def register():
    """Register the RBLN platform."""
    return "vllm_rbln.platform.RblnPlatform"
```

## 2. vLLM이 플러그인을 불러오고 플랫폼을 선택한다

`current_platform`을 처음 참조하면 vLLM은 설치된 패키지들에서 플랫폼 entry point를 읽는다. `VLLM_PLUGINS`를 설정했다면 목록에 있는 이름만 불러온다. 아래 vLLM 코드 블록은 흐름을 보여주기 위해 로그와 일부 예외 처리를 생략했다.

```python
from importlib.metadata import entry_points

allowed_plugins = envs.VLLM_PLUGINS
platform_plugins = {}

for plugin in entry_points(group="vllm.platform_plugins"):
    if allowed_plugins is None or plugin.name in allowed_plugins:
        platform_plugins[plugin.name] = plugin.load()
```

여기서 `plugin.load()`가 불러오는 건 `register` 함수 자체다. 다음 단계에서 vLLM은 내장 플랫폼 함수와 외부 플러그인 함수를 각각 호출한다. 반환값이 `None`이 아니면 활성 후보로 기록한다.

```python
from itertools import chain

if envs.VLLM_TARGET_DEVICE == "cpu":
    return cpu_platform_plugin()

activated_plugins = []
for name, func in chain(builtin_platform_plugins.items(), platform_plugins.items()):
    if func() is not None:
        activated_plugins.append(name)
```

예를 들어 CUDA의 내장 함수는 일반적인 환경에서 GPU가 있는지 확인한다. RBLN의 `register()`는 NPU 조회 없이 클래스 경로를 반환하므로, 함수가 정상적으로 불려오고 호출되면 RBLN은 활성 후보가 된다.

```python
activated_external = [name for name in activated_plugins if name in platform_plugins]

if len(activated_external) > 1:
    raise RuntimeError("Only one platform plugin can be activated")
if len(activated_external) == 1:
    return platform_plugins[activated_external[0]]()

# 활성 외부 후보가 없으면 내장 후보를 선택한다.
```

CUDA와 RBLN이 모두 활성 후보여도 RBLN이 유일한 외부 후보라면 RBLN이 우선한다. `VLLM_TARGET_DEVICE=cpu`를 명시하면 위 코드의 첫 분기에서 바로 CPU 플랫폼을 반환한다.

## 3. 선택한 플랫폼의 메서드를 vLLM 공통 코드가 호출한다

vLLM은 선택된 클래스의 경로로 클래스를 가져와 인스턴스를 만들고 보관한다.

```python
if _current_platform is None:
    class_path = resolve_current_platform_cls_qualname()
    _current_platform = resolve_obj_by_qualname(class_path)()

return _current_platform
```

이후 vLLM 설정 코드에서 다음 호출을 하면, `current_platform`이 가리키는 클래스의 구현이 실행된다.

```python
current_platform.check_and_update_config(self)
```

`RblnPlatform`이 선택됐다면 이 호출은 RBLN 쪽 설정 메서드를 실행한다. vLLM은 이때 이미 만들어 둔 플랫폼 객체를 사용한다.

## 위임 예시: `execute_model()`에서 모델의 `forward()`까지

`check_and_update_config()`가 플랫폼 객체에 설정을 맡기는 호출이라면, `execute_model()`은 워커 객체에 모델 실행을 맡기는 호출이다. vLLM 본체의 워커 래퍼는 다음과 같이 호출한다.

```python
return self.worker.execute_model(scheduler_output)
```

워커가 자동 설정이라면 CUDA 플랫폼은 GPU 워커를, RBLN 플랫폼의 `vllm` 모델 경로는 `RBLNWorker`를 지정한다. 공통 호출 코드는 같고, `self.worker`에 실제로 들어 있는 객체에 따라 실행되는 메서드가 달라진다.

`RBLNWorker.execute_model()`은 다시 모델 runner에 실행을 맡긴다.

```python
output = self.model_runner.execute_model(scheduler_output, intermediate_tensors)
```

runner는 실행에 필요한 입력을 준비하고 `self.model_executable(...)`을 호출한다. 그 실행 대상의 기반이 되는 `model_wrapper` 안에는 다음과 같은 모델 호출이 있다.

```python
model_output = self.model(
    input_ids=input_ids,
    positions=positions,
    intermediate_tensors=intermediate_tensors,
    inputs_embeds=inputs_embeds,
    **kwargs,
)
```

PyTorch 모듈을 `self.model(...)`처럼 호출하면 모델의 `forward()`로 이어진다. 이 예시에서 vLLM 본체가 실행을 위임하는 지점은 `self.worker.execute_model(...)`이고, 실제 모델 호출은 RBLN runner 쪽에 있다.

소스 기준은 vLLM `main`의 `2c1fc9b`와 vllm-rbln `dev`의 `9df6991`이다.
