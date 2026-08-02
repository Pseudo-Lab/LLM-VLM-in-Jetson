# Qwen3-4B AWQ 챗봇 서빙 가이드

직접 구현한 AWQ INT4 양자화 모델을 **vLLM으로 서빙**하고 **채팅**까지 해보는 과정입니다.
양자화 파이프라인 자체는 [AWQ_USAGE.md](AWQ_USAGE.md), 구현 원리·실험 결과는
[DESIGN_NOTES.md](DESIGN_NOTES.md)를 참고하세요.

## 개요

본 문서는 직접 구현한 AWQ INT4 양자화 모델을 실제 추론 환경에 배포하고 검증하는 절차를 안내합니다.
대상 모델은 Qwen3-4B-Instruct를 기반으로 양자화한 **`skt0725/qwen3-4b-instruct-awq-ko`** 이며,
추론 서버로는 **vLLM**을 사용합니다.

절차는 다음 세 단계로 구성됩니다.

1. Hugging Face Hub에서 양자화 모델을 내려받습니다.
2. vLLM을 통해 OpenAI 호환 API 서버로 모델을 서빙합니다.
3. 해당 API에 요청을 보내 대화형 추론을 검증합니다.

양자화를 통해 모델 크기는 FP16 기준 8.04GB에서 INT4 기준 2.67GB로 약 3배 감소하였으며,
KMMLU 정확도는 45.81%에서 42.98%로 성능 저하를 최소화하였습니다.

---

## 사전 요구사항 (중요)

> **GPU(CUDA)가 반드시 필요합니다.** AWQ 양자화 모델은 vLLM의 CUDA 커널(`awq_marlin` 등)로만
> 돌아갑니다. **CPU-only 환경에서는 서빙되지 않습니다.**

- NVIDIA GPU (Ampere 이상 권장 — RTX 3090/4090, A100 등)
- CUDA 지원 환경 (RunPod GPU 파드, 로컬 GPU 서버 등)
- Python 3.10+
- VRAM: 모델 2.67GB + KV 캐시. 8GB 이상이면 충분, 여유 있으면 `--max-model-len`를 키울 수 있음

확인:
```bash
nvidia-smi --query-gpu=name,memory.total --format=csv
```
GPU 이름이 나오지 않으면 GPU 환경이 아닙니다.

---

## 1. 설치

```bash
# huggingface CLI (모델 다운로드용)
pip install -U "huggingface_hub[cli]"

# vLLM (서빙 엔진) — 몇 분 걸립니다
pip install vllm

# 채팅 스크립트용
pip install requests
```

확인:
```bash
vllm --version
```

---

## 2. 모델 다운로드

Hugging Face에서 양자화된 모델을 받습니다:

```bash
HF_HUB_DISABLE_XET=1 hf download skt0725/qwen3-4b-instruct-awq-ko \
  --local-dir ./outputs/qwen3-4b-instruct-awq-ko
```

- `HF_HUB_DISABLE_XET=1` : 일부 환경에서 다운로드가 멈추는 Xet 전송을 끄고 안정적인 HTTPS로 받습니다.
- 크기 약 2.7GB, 파일 6개 (safetensors 본체 + config/tokenizer 등).

> 채팅봇이 아니라 base 모델(토큰 이어쓰기)을 원하면 `skt0725/qwen3-4b-awq-ko`를 받으세요.
> 단, base는 chat_template이 없어 대화용으로는 부적합합니다.

---

## 3. vLLM 서버 실행

**터미널 A** (서버 전용 — 계속 켜둡니다):

```bash
vllm serve ./outputs/qwen3-4b-instruct-awq-ko \
  --served-model-name qwen3-awq \
  --quantization awq_marlin \
  --max-model-len 4096 \
  --port 8000 --host 0.0.0.0
```

옵션 설명:
- `--served-model-name qwen3-awq` : API에서 부를 짧은 별칭 (실체는 위 경로의 우리 AWQ 모델)
- `--quantization awq_marlin` : Ampere 이상에서 빠른 AWQ 커널. 문제 시 `awq`로 대체
- `--max-model-len 4096` : 최대 컨텍스트 길이. VRAM이 빠듯하면 줄이세요

다음 로그가 뜨면 준비 완료:
```
INFO: Application startup complete.
INFO: Uvicorn running on http://0.0.0.0:8000
```

서버 상태 확인 (다른 터미널):
```bash
curl -s http://localhost:8000/v1/models
```
`"root":"./outputs/qwen3-4b-instruct-awq-ko"`가 보이면 우리 모델이 정상 로드된 것입니다.

---

## 4. 채팅

### 방법 A: 단발 테스트 (curl)

```bash
curl http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"model":"qwen3-awq","messages":[{"role":"user","content":"대한민국의 수도는 어디야?"}],"max_tokens":100}'
```

### 방법 B: 대화형 스크립트 (chat.py)

아래 내용으로 `chat.py`를 만듭니다:

```python
import requests

URL = "http://localhost:8000/v1/chat/completions"
history = []

print("채팅 시작 (종료: quit 입력 또는 Ctrl+C)\n")
while True:
    try:
        user = input("나: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\n종료"); break
    if user in ("quit", "exit"):
        break
    history.append({"role": "user", "content": user})
    try:
        r = requests.post(URL, json={
            "model": "qwen3-awq",
            "messages": history,
            "max_tokens": 512,
            "temperature": 0.7,
        }, timeout=120)
        data = r.json()
        if "choices" not in data:
            print("⚠️ 서버 응답 이상:", data, "\n")
            history.pop()
            continue
        reply = data["choices"][0]["message"]["content"]
        print("봇:", reply, "\n")
        history.append({"role": "assistant", "content": reply})
    except Exception as e:
        print("⚠️ 요청 실패:", e, "\n")
        history.pop()
```

**터미널 B**에서 실행:
```bash
python3 chat.py
```

`history` 리스트로 이전 대화를 함께 보내므로 문맥이 유지됩니다. 종료는 `quit` 또는 Ctrl+C.

---

## 문제 해결 (Troubleshooting)

| 증상 | 원인 / 해결 |
|------|------|
| `Connection refused` (채팅 시) | 서버가 아직 안 떴거나 로딩 중. `curl http://localhost:8000/v1/models`로 준비 확인 후 재시도 |
| `CUDA error: out of memory` | VRAM 부족. `--max-model-len` 축소, `--gpu-memory-utilization 0.80`, `--swap-space 0` |
| `No available memory for the cache blocks` | KV 캐시 자리 부족. `--gpu-memory-utilization`를 **올리고** `--max-model-len`를 낮추세요 |
| `awq_marlin` 관련 커널 에러 | `--quantization awq`로 변경 |
| `TextEncodeInput must be Union[...]` (특정 입력에서만) | 토크나이저 라이브러리 엣지 케이스. 모델·양자화와 무관. `pip install -U tokenizers transformers`로 완화 가능 |
| `hf: command not found` | `pip install -U "huggingface_hub[cli]"` |
| CPU-only 환경 | AWQ + vLLM은 GPU 필수. GPU 환경으로 이동 |

---

## Jetson Orin Nano 등 엣지 디바이스 참고

Jetson(aarch64 + JetPack)에서는 `pip install vllm`이 안 됩니다. dusty-nv의
[jetson-containers](https://github.com/dusty-nv/jetson-containers)로 vLLM 컨테이너를 써야 합니다:

```bash
jetson-containers run \
  -v /path/to/models:/models -p 8000:8000 \
  -e VLLM_USE_FLASHINFER_SAMPLER=0 \
  $(autotag vllm) \
  vllm serve /models/qwen3-4b-instruct-awq-ko \
    --served-model-name qwen3-awq --quantization awq_marlin \
    --max-model-len 1024 --gpu-memory-utilization 0.85 \
    --swap-space 0 --enforce-eager --port 8000 --host 0.0.0.0
```

Orin Nano 8GB 주의사항:
- GPU/CPU가 물리 RAM을 공유하므로 데스크톱 GUI·VS Code 원격 서버 등을 끄고 RAM을 5GB+ 확보
  (`sudo systemctl isolate multi-user.target`, `pkill -f vscode-server`)
- `--enforce-eager`로 CUDA 그래프 캡처를 꺼 메모리 절약
- NVML 미지원으로 FlashInfer 샘플러가 죽으므로 `-e VLLM_USE_FLASHINFER_SAMPLER=0` 필요
- 일부 transformers 버전에서 `extra_special_tokens`(list) 로딩 실패 시, 해당 필드를 `{}`로 수정

> 이 문제들 때문에 학습·데모 목적이라면 **GPU 클라우드(RunPod 등)에서 서빙하는 것을 권장**합니다.
