# AWQ 양자화 모듈 사용 가이드

AWQ INT4 양자화를 직접 구현한 모듈. `AutoModelForCausalLM`으로 로드되는 모든 HF 모델에
적용 가능하며, gptqmodel/vLLM이 바로 로드하는 AWQ 표준 포맷으로 저장합니다.

## 빠른 시작

```bash
python run_awq.py --model Qwen/Qwen3-4B --calib-data pileval

# 다른 모델도 동일한 커맨드 (Llama, Mistral, Phi, TinyLlama 등)
python run_awq.py --model meta-llama/Llama-3.2-3B --calib-data wikitext2
python run_awq.py --model mistralai/Mistral-7B-v0.3 --calib-data c4
python run_awq.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --calib-data wikitext2

# 한국어 calibration
python run_awq.py --model Qwen/Qwen3-4B --calib-data kowikitext
```

```python
# Python API
from awq import AWQQuantizer
quantizer = AWQQuantizer("Qwen/Qwen3-4B", w_bit=4, group_size=128)
quantizer.quantize(calib_data="pileval", output_dir="./outputs/qwen3-4b-awq")
```

파이프라인: **calibration**(activation 통계 수집, hook 기반) → **scale 탐색**(grid search)
→ **INT4 양자화** → **export**(safetensors). `lm_head`는 자동 제외.

## 주요 옵션

| 인자 | 기본값 | 설명 |
|------|--------|------|
| `--model` | `Qwen/Qwen3-4B` | HF 모델 ID / 로컬 경로 |
| `--calib-data` | `pileval` | `pileval` / `wikitext2` / `kowikitext`(한국어) / `c4` |
| `--n-samples` / `--seq-len` | 128 / 512 | calibration 규모 (코드 검증만 할 땐 16 / 256) |
| `--w-bit` / `--group-size` | 4 / 128 | 양자화 설정 |
| `--skip-layers` | `lm_head` | 제외 레이어 (정확한 모듈 이름) |

## 대상 모델 요건

- `in_features % group_size == 0` — 안 맞는 레이어는 자동 스킵(FP16 유지)
- `out_features % 8 == 0` — INT4 8개 → INT32 패킹 단위
- gated repo(Gemma, Llama 등)는 HF 접근 승인 + 로그인 필요

커스텀 calibration 데이터는 `awq/calibration.py`의 `get_calib_dataset()`에 분기 추가
(텍스트 리스트만 만들면 토크나이즈/분할은 공통 처리).

## 출력 포맷 (AWQ GEMM 표준)

| 텐서 | Shape |
|------|-------|
| `qweight` | `[in_features, out_features // 8]` INT32 |
| `scales` | `[n_groups, out_features]` FP16 |
| `qzeros` | `[n_groups, out_features // 8]` INT32 |

INT32 내부 패킹은 **인터리브 순서 `[0, 2, 4, 6, 1, 3, 5, 7]`** (순차로 하면 출력이 깨짐).

로드 (GPU 필요):

```python
# transformers + gptqmodel — config의 quantization_config로 자동 인식
model = AutoModelForCausalLM.from_pretrained(path, torch_dtype="float16", device_map="auto")

# vLLM
llm = LLM(model=path, quantization="awq")
```

## 검증 절차

1. 작은 모델로 스모크 테스트: `--model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --n-samples 16 --seq-len 256`
2. generate 테스트 — 깨진 토큰 반복이면 패킹 문제, 품질만 낮으면 calibration 문제
3. `lm_eval` 벤치마크 — 랜덤 수준(~25%)이면 export 버그 의심:
   ```bash
   python -m lm_eval --model hf \
     --model_args pretrained=<output_dir>,dtype=float16,device_map=auto \
     --tasks kmmlu --batch_size 16
   ```

## 자주 겪는 문제

- **`autoawq` 이름 충돌**: `run_awq.py`가 importlib로 로컬 `awq/`를 명시적 로드해 회피
- **Colab pip install 후 ImportError**: 설치 후 반드시 런타임 재시작
- **출력이 깨진 토큰 반복**: 패킹 인터리브 순서 확인
