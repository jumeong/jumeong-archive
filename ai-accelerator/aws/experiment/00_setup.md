아래와 같이 Worker Node 생성 완료.
```bash
ubuntu@ip-10-0-1-194:~/workshop$ kubectl get nodes -o wide
NAME                                      STATUS   ROLES    AGE     VERSION                INTERNAL-IP   EXTERNAL-IP     OS-IMAGE                        KERNEL-VERSION                     CONTAINER-RUNTIME
ip-10-0-2-46.us-west-2.compute.internal   Ready    <none>   2m55s   v1.33.13-eks-cb19647   10.0.2.46     44.251.247.49   Amazon Linux 2023.12.20260831   6.12.103-127.188.amzn2023.x86_64   containerd://2.2.5+unknown
```

아래처럼 Node 정보에서 NeuronCore를 확인할 수 있다.
```bash
Capacity:
  aws.amazon.com/neuron:      1
  aws.amazon.com/neuroncore:  2
  cpu:                        8
```

Worker Node에는 AWS에서 제공하는 Session Manager를 통해 접속했고 `neuron-ls`와 같은 명령어를 실행해봄.
```bash
[ec2-user@ip-10-0-2-46 ~]$ neuron-ls
instance-type: trn1.2xlarge
instance-id: i-0ef32af9c82e2a5c4
+--------+--------+----------+--------+--------------+----------+------+
| NEURON | NEURON |  NEURON  | NEURON |     PCI      |   CPU    | NUMA |
| DEVICE | CORES  | CORE IDS | MEMORY |     BDF      | AFFINITY | NODE |
+--------+--------+----------+--------+--------------+----------+------+
| 0      | 2      | 0-1      | 32 GB  | 0000:00:1e.0 | 0-7      | -1   |
+--------+--------+----------+--------+--------------+----------+------+
```

vLLM deployment까지 실행해본 후, vLLM 관련 환경설정 출력.
```bash
ubuntu@ip-10-0-1-194:~/workshop$ kubectl exec -it deploy/vllm-deployment -c vllm-server -- env | grep -E 'NEURON|VLLM|MAX|TENSOR'
VLLM_TARGET_DEVICE=neuron
NEURON_LOGICAL_NC_CONFIG=1
NEURON_COMPILE_CACHE_URL=/shared/model/cache
MAX_MODEL_LEN=1024
MAX_NUM_SEQS=4
NEURON_RT_ASYNC_EXEC_MAX_INFLIGHT_REQUESTS=4
NEURON_RT_VISIBLE_CORES=0-1
TENSOR_PARALLEL_SIZE=2
VLLM_NEURON_FRAMEWORK=neuronx-distributed-inference
NEURON_COMPILED_ARTIFACTS=/shared/model/cache
NEURON_RT_LOG_LEVEL=ERROR
```

Worker Node에서 직접 vLLM 실행 여부를 확인해본 모습.
```bash
[ec2-user@ip-10-0-2-46 bin]$ ps -ef | grep -i vllm
ec2-user   26029   26000  0 16:28 ?        00:00:00 /mountpoint-s3/bin/mount-s3 ai-infra-summit-vllm-models-cache-727641133746 /dev/fd/3 --allow-root --foreground --user-agent-prefix=s3-csi-driver/2.8.0 credential-source#driver k8s/v1.33.13-eks-4cc7921 md/install#helm
root       27708   26158  3 16:32 ?        00:00:08 python -m vllm.entrypoints.openai.api_server --model=tinyLlama/TinyLlama-1.1B-Chat-v1.0 --max-num-seqs=4 --max-model-len=1024 --tensor-parallel-size=2 --port=8080 --device=neuron --override-neuron-config={"enable_bucketing":false}
```