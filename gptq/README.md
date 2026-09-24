# Llama 3.2 11B Vision 4bit 경량화 (GPTQ / NF4)

## 핵심 결과 요약

| 구성 | 한국어 PPL (Δ vs fp16) | K-DTCBench | Peak VRAM |
|------|------------------------|-----------|-----------|
| fp16 (원본) | 8.57 | 29.2% | ~22GB |
| GPTQ 4bit (한국어 혼합 캘리브 70/30) | 9.14 (+6.6%) | 30.4% | ~11GB |
| **NF4 전체 4bit (비전 포함)** | 9.23 (+7.7%) | 37.9% | **8.15GB** |
| NF4 + lm_head 4bit + 타일2 | 9.33 (+8.9%) | 32.5% | **6.88GB** |

- GPTQ 캘리브레이션은 **한국어/영어 혼합(70/30)** 데이터를 사용 — 캘리브 데이터의 언어 구성이 한국어 품질에 직접 영향을 줌
- K-DTCBench(240문제) 점수는 전 구성이 랜덤(25%) 부근 노이즈 — 이 모델의 한국어 문서 이해 한계이며, 양자화로 인한 추가 손상은 없다는 것까지만 해석.

## NF4 전체 4bit 

GPTQ(GPTQModel)는 mllama에서 **텍스트 레이어만** 양자화할 수 있음

NF4(NormalFloat4)는 bitsandbytes 라이브러리의 4bit 포맷(QLoRA 논문에서 도입). 신경망 가중치가 대체로 정규분포를 따른다는 점을 이용해, 정규분포에 맞게 배치한 16개의 대표값으로 가중치를 반올림한다. GPTQ와의 핵심 차이 두 가지:

1. **캘리브레이션이 필요 없음** — GPTQ는 데이터를 흘려보내며 가중치를 보정하지만, NF4는 데이터 없이 즉석 변환. 그래서 캘리브 데이터의 언어 편향 문제 자체가 없음.
2. **모델 전체에 적용됨** — 텍스트/비전 구분 없이 모든 Linear 레이어를 4bit로 적재할 수 있음.

**어떻게 적용했나.** 별도 양자화 단계 없이, 모델 로드 시 `BitsAndBytesConfig`를 넘기면 끝이라고 함..:

```python
from transformers import BitsAndBytesConfig, MllamaForConditionalGeneration

bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,   # 양자화 상수까지 한 번 더 압축
)
model = MllamaForConditionalGeneration.from_pretrained(path, quantization_config=bnb, device_map="auto")
```

**결과.** 비전 인코더·cross-attn 포함 전체가 4bit로 적재되어 peak VRAM 11GB → **8.15GB**. 품질은 캘리브 없이도 GPTQ 한국어 캘리브와 동급(위 요약표). 여기에 lm_head 4bit와 타일 축소를 더해 **6.88GB**까지 확인(아래 표).

## 추가 압축 실험 (NF4 기준)

lm_head 4bit 양자화(`nf4_lmhead`)와 이미지 타일 수 축소(`--max-tiles`)를 조합한 결과 (K-DTCBench 정확도 / peak VRAM):

| | 타일 4 (기본) | 타일 2 | 타일 1 |
|---|---|---|---|
| nf4 | 37.9% / 8.15GB | 35.0% / 7.66GB | 23.3% / 7.44GB |
| nf4_lmhead | 35.0% / 7.37GB | **32.5% / 6.88GB** | 23.3% / 6.66GB |

- **타일 1은 붕괴**(23.3%, 랜덤 이하) — 문서 읽기 불가, 채택 불가.
- lm_head 4bit·타일 2는 각각 −2.9pt 수준으로 노이즈 범위 내 = 실질 무해. 속도 보너스 306s→224s.
- **현실적 최선: `nf4_lmhead` + 타일 2 = 6.88GB.** 최대 압축(타일 1)도 6.66GB로 6.5GB 목표엔 미달 — 나머지 ~0.4GB는 KV캐시/활성값/경량 프루닝 몫.


## 환경

| 항목 | 값 |
|------|-----|
| GPU | NVIDIA RTX 3080 Ti, 12GB |
| CUDA | 12.6 (드라이버 560.94) |
| 시스템 RAM | 64GB (CPU 오프로드용) |
| Python | **3.11 권장** (3.13은 gptqmodel/torch 휠 미지원 가능성) |

> 12GB VRAM으로 11B fp16(~22GB)은 못 올리지만, GPTQModel이 레이어 단위로 양자화하며 나머지를 RAM으로 오프로드하므로 동작

## 설치

```powershell
# Python 3.11 가상환경 생성
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1

# 나머지 먼저 → CUDA torch는 "맨 마지막"에 설치
# (gptqmodel 7.x가 torch>=2.8 CPU 빌드를 끌어오므로, CUDA 빌드로 마지막에 덮어써야 함)
pip install --upgrade pip
pip install -r requirements.txt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126 --force-reinstall
```

> `pip install` 순서를 바꾸면 torch가 CPU 빌드로 덮여 `torch.cuda.is_available()==False`가 되니 주의.

## 모델 접근 (필수)

`meta-llama/Llama-3.2-11B-Vision-Instruct`는 **gated 모델**.

1. https://huggingface.co/meta-llama/Llama-3.2-11B-Vision-Instruct 에서 라이선스 승인
2. HF 액세스 토큰 발급 후 `.env`에 설정 (`.env.example` 복사)

## 실행

### 양자화 파이프라인 (GPTQ)

```powershell
python src/download_model.py     # 1. 원본 모델 다운로드
python src/quantize.py           # 2. GPTQ 4비트 양자화 (한국어 혼합 캘리브)
python src/evaluate.py           # 3. 전/후 비교
```

세부 설정은 `configs/gptq_config.yaml`. NF4는 별도 양자화 단계가 없다 — 평가 스크립트가 로드 시 즉석 양자화 진행

### 평가

```powershell
# K-DTCBench (한국어 문서/표/차트 VQA, 240문제)
python src/eval_kdtcbench.py --model nf4_lmhead --max-tiles 2
#   --model  fp16 | gptq | nf4 | nf4_lmhead
#   --max-tiles N  이미지 타일 상한 축소 (기본 4)
#   --prune-k K    Block Influence 순 self-attn K층 드롭 (실험용)

# Perplexity (영어/한국어)
python src/eval_ppl.py --model nf4_lmhead
```

결과는 `results/*.json`으로 저장.

