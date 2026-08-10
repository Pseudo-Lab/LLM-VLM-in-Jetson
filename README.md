# ShortGPT Depth Pruning 사용법

Transformer의 **레이어(깊이) 단위 structured 프루닝** 구현입니다. 잉여 레이어를 통째로 제거합니다. **레이어 개수 자체를 줄여** llama.cpp 같은 dense 런타임에서 메모리·속도가 실제로 줄어듭니다.

<details>
<summary>dense 런타임?</summary>

**dense 런타임** = 가중치를 **dense tensor 그대로 저장·계산**하는 실행 환경 (llama.cpp / GGUF 등). 0인 값도 그대로 저장하고 곱합니다.

프루닝 종류에 따라 효과가 갈립니다:

| 프루닝 종류 | 하는 일 | dense 런타임에서 |
|---|---|---|
| **Unstructured (마스킹, 예: Wanda)** | 일부 가중치를 **0으로** | 텐서 크기 그대로 → **이득 없음** (0도 저장·계산). sparse 런타임에서만 0을 건너뛰어 이득 |
| **Structured (ShortGPT depth, width)** | **레이어/차원을 통째로 제거** | 텐서가 **물리적으로 작아짐** → 메모리·속도 **실제 감소** |

즉 배포 런타임이 dense(ex. llama.cpp on Jetson)라면, **레이어를 진짜로 없애는 structured 프루닝** 일 경우 메모리가 줄어듭니다.
</details>

---

## 1. 원리 (Block Influence)

각 레이어 `i`의 **Block Influence(BI)** 를 보정 데이터로 측정합니다:

```
BI_i = 1 - mean_cosine(레이어 입력 hidden, 레이어 출력 hidden)
```

- 입력 ≈ 출력 (코사인 ≈ 1) → 레이어가 잔차 스트림을 **거의 안 바꿈** → 잉여 → **BI 낮음 → 제거 대상**
- 입력 ≠ 출력 → 레이어가 **열심히 일함** → BI 높음 → 보존

→ BI가 낮은 레이어부터 `ratio` 만큼 제거하고, **남는 레이어의 원래 순서는 보존**합니다.
(ShortGPT 논문: arXiv 2403.03853)

---

## 2. shortGPT 적용 가능 모델

이 구현은 **`model.model.layers` 가 transformer 레이어 리스트(`nn.ModuleList`)인 표준 HF decoder** 를 가정합니다. (ex. Llama, Qwen, Mistral, Gemma)

단, MoE 혹은 Mamba 계열의 경우 레이어를 통으로 날리는 shortGPT에는 적합하지 않습니다. 예를 들어, MoE의 경우 Expert가 모여있는 레이어를 날릴 수도 있습니다.

## 3. calibration

- `Block Influence`를 구하기 위해 레이어 입력과 출력을 구해야하는데, 이는 실제 데이터를 넣고 forward를 해야 나옵니다.
- 따라서 calibration 데이터로 forward를 돌려야 레이어별 활성값을 얻고, 그걸로 `BI`를 구합니다.
- calibration data는 사용할 메인 언어로 지정하면 됩니다.
- `calibration.datasets` 에는 **아무 HF 데이터셋**이나 넣을 수 있습니다 (`text`/`instruction`/`question`/`output` 등 필드를 자동 추출)
- calibration dataset은 한국어 BI를 측정하는 용도이기 때문에, 어느 것을 사용하든 상관없습니다.

---

## 4. 파일 구성

`run_depth_prune.py`가 이들을 import

| 파일 | 역할 |
|---|---|
| `shortgpt/scripts/run_depth_prune.py` | 실행 진입점 (yaml → 로드→측정→제거→저장) |
| `shortgpt/configs/prune_phi4.yaml` | 설정 예시 |
| `shortgpt/src/prune/depth_prune.py` | **핵심**: `compute_block_influence()` + `prune_depth()` |
| `shortgpt/src/prune/calibration.py` | 보정 데이터 로드 + 토크나이즈 |
| `shortgpt/src/prune/report.py` | 결과 리포트 출력 (run script가 import) |
| `shortgpt/src/common/model_loader.py` | 모델 로딩 (run script가 import) |
| `shortgpt/src/common/param_stats.py` | 파라미터 카운트 (`depth_prune.py`가 import) |


---

## 5. 환경

**uv**

```bash
cd shortgpt
uv sync                      

# 실행: uv run python scripts/...  또는 .venv/bin/python scripts/...
```

**pip**

```bash
cd shortgpt
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt   
```
---

## 6. 실행 방법

```bash
cd shortgpt
.venv/bin/python scripts/run_depth_prune.py --config configs/prune_phi4.yaml
# 출력 디렉토리를 바꾸려면:
.venv/bin/python scripts/run_depth_prune.py --config configs/prune_phi4.yaml --out artifacts/my-pruned
```

### 설정 파일 (`configs/prune_phi4.yaml`)
```yaml
model:
  name: microsoft/phi-4        # HF 모델 ID 또는 로컬 경로
  dtype: bfloat16
prune:
  ratio: 0.30                  # 제거할 레이어 비율 (0.30 = 레이어의 30% 제거, 예: 40층 → 28층). 논문의 대표 결과는 LLaMA2에서 25~27%
calibration:
  datasets:                    # BI 측정용 한국어 보정 텍스트 (HF 데이터셋)
    - MarkrAI/KoCommercial-Dataset
    - beomi/KoAlpaca-RealQA
    - beomi/kowikitext-qa-ref-detail-preview
  seq_len: 1024                # 보정 시퀀스 길이. 논문값
  n_samples: 256               # 보정 샘플 수 (많을수록 BI 안정적, 메모리 사용량은 증가). 임의값
  seed: 42
output:
  dir: artifacts/phi4-pruned   # 결과 저장 경로 (--out 으로 덮어쓰기 가능)
```

### 실행하면 보이는 출력
```
Block Influence per layer: [0.44, 0.18, 0.13, ... 0.05]   # 레이어별 BI (낮을수록 잉여)
Layers kept: [0, 1, 2, ..., 26, 38]                       # 남긴 레이어 인덱스
=== Pruning Report ===
params: 14,659,...  ->  10,569,...  (27.9% 감축)
메모리(4bit 추정): 7.33GB -> 5.28GB
```
→ 결과 모델은 `output.dir` 에 표준 HF 포맷(`model.safetensors`+`config.json`+토크나이저)으로 저장됩니다. 그대로 `AutoModelForCausalLM.from_pretrained()` 로 로드/평가/서빙 가능.

---

## 7. 코드 사용

스크립트 없이 함수만 가져다 쓸 수도 있습니다:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.prune.calibration import load_korean_texts, tokenize_texts
from src.prune.depth_prune import compute_block_influence, prune_depth

model = AutoModelForCausalLM.from_pretrained("microsoft/phi-4", torch_dtype=torch.bfloat16, device_map="cuda")
tok = AutoTokenizer.from_pretrained("microsoft/phi-4")

# 1) 보정 데이터 준비
texts = load_korean_texts(["beomi/kowikitext-qa-ref-detail-preview"], n_samples=128, seed=42)
input_ids, attn = tokenize_texts(texts, tok, seq_len=1024, return_mask=True)
batch = {"input_ids": input_ids.cuda(), "attention_mask": attn.cuda()}

# 2) Block Influence 측정
bi = compute_block_influence(model, [batch])   #  레이어별 텐서

# 3) 낮은 BI 레이어 30% 제거
model, info = prune_depth(model, ratio=0.30, bi_scores=bi)
print(info["layers_kept"], info["ratio_actual"])

# 4) 저장
model.save_pretrained("my-pruned"); tok.save_pretrained("my-pruned")
```

---


