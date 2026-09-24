# FOEM 폴더 사용법

Mistral3 계열 VLM(`Ministral-3-3B-Instruct-2512-BF16` 등)을 **GPTQ**와 **FOEM**(First-Order Error Matters, AAAI 2026) 두 가지 방식으로 양자화하고, WikiText-2 PPL / KMMLU 정확도로 품질을 비교하는 실험 코드 모음.

## 디렉토리 구성

```
FOEM/
├── pseudo/                                    # 실행 스크립트 (핵심 코드)
│   ├── quantize_mistral.py                    # 양자화 + PPL/KMMLU 평가 + README 자동 생성 (메인 엔트리)
│   ├── eval_ppl.py                            # WikiText-2 PPL 단독 평가
│   ├── eval_kmmlu.py                          # KMMLU(45개 과목) 단독 평가
│   ├── collect_base_examples.py               # 보고서용: BF16 베이스 모델 예시 문항 4개 확률분포 수집
│   ├── collect_extra_examples.py              # 보고서용: BF16/GPTQ/FOEM 3-way 예시 문항 4개 확률분포 수집
│   ├── update_report_with_base.py             # base_kmmlu.log 를 파싱해 KMMLU_REPORT.md 재생성
│   └── KMMLU_REPORT.md                        # 생성된 비교 보고서 (BF16 vs GPTQ 4bit vs FOEM 3bit)
├── Ministral-3-3B-Instruct-2512-BF16_gptq_4bit/  # quantize_mistral.py 로 생성된 GPTQ 4-bit 산출물
├── Ministral-3-3B-Instruct-2512-BF16_foem_3bit/  # quantize_mistral.py 로 생성된 FOEM 3-bit 산출물
├── logs/                                      # 각 스크립트 실행 로그 (git에는 추적하지 않음, .gitignore 처리)
└── README.md                                  # 가짜연구소(Pseudo-Lab) 프로젝트 템플릿 README
```

각 양자화 산출물 디렉토리(`*_gptq_4bit`, `*_foem_3bit`)는 `model.safetensors`, `config.json`, `tokenizer.json` 등 HuggingFace 로딩에 필요한 파일과, 양자화 통계·PPL·KMMLU 결과가 정리된 자체 `README.md`를 담고 있다.

## 환경 요구사항

- CUDA GPU (스크립트 내 `torch.cuda` 사용 전제, Jetson/서버 등 CUDA 환경)
- Python 패키지: `torch`, `transformers`, `datasets`, `huggingface_hub`, `gptqmodel` (FOEM 사용 시 `FOEMConfig`를 포함한 버전 필요 → `pip install -U gptqmodel`)
- 원본 모델은 사전에 `huggingface-cli download` 등으로 로컬 캐시에 받아둬야 함 (`snapshot_download(..., local_files_only=True)`로 로드하므로 최초 1회는 온라인 다운로드 필요)
- `quantize_mistral.py`의 출력 경로는 `/workspace/LLM-VLM-in-Jetson`으로 하드코딩되어 있음(`--out`으로 덮어쓰기 가능). `collect_base_examples.py`/`collect_extra_examples.py`/`update_report_with_base.py`의 모델·로그 경로도 하드코딩되어 있으니 다른 환경에서 실행 시 스크립트 상단 경로 상수를 수정해야 함.

## 실행 방법

모든 스크립트는 `FOEM/pseudo/` 안에서 실행한다 (`quantize_mistral.py`가 같은 디렉토리의 `eval_kmmlu.py`를 직접 import 하기 때문).

```bash
cd FOEM/pseudo
```

### 1. 양자화 (메인 엔트리) — `quantize_mistral.py`

GPTQ와 FOEM을 동일 코드로 산출한다. 차이는 `QuantizeConfig`에 `foem=` 인자 유무뿐. 양자화 후 자동으로 PPL(WikiText-2)과 KMMLU(45개 과목, 5-shot) 평가까지 수행하고, 결과를 산출물 디렉토리의 `README.md`에 기록한다.

```bash
# GPTQ 4-bit (기본 모델: mistralai/Mistral-Small-3.1-24B-Instruct-2503)
python quantize_mistral.py --method gptq --bits 4

# FOEM 4-bit
python quantize_mistral.py --method foem --bits 4

# FOEM 3-bit, 모델 지정
python quantize_mistral.py --method foem --bits 3 --model mistralai/Ministral-3-3B-Instruct-2512-BF16
```

주요 옵션:

| 옵션 | 기본값 | 설명 |
|---|---|---|
| `--method` | (필수) | `gptq` 또는 `foem` |
| `--bits` | (필수) | `3` 또는 `4` |
| `--model` | `mistralai/Mistral-Small-3.1-24B-Instruct-2503` | 원본 모델 (로컬 캐시에 있어야 함) |
| `--nsamples` | 256 | 캘리브레이션 샘플 수 (allenai/c4, 실패 시 wikitext-2로 폴백) |
| `--group-size` | 128 | 양자화 그룹 크기 |
| `--alpha` / `--beta` | 0.0 / 0.2 | FOEM 하이퍼파라미터 (alpha=0 → 1차 보정 비활성, beta → 오차 피드백 강도) |
| `--attn` | `eager` | attention 구현 (`sdpa`는 transformers 5.x에서 meta-tensor 버그로 비권장) |
| `--offload-disk` | False | 디스크 오프로드 (켜면 embed_tokens 출력이 meta tensor가 되어 깨질 수 있음) |
| `--out` | `/workspace/LLM-VLM-in-Jetson/{모델명}_{method}_{bits}bit` | 저장 경로 |
| `--skip-ppl` / `--skip-kmmlu` | False | 시간 절약용 평가 생략 옵션 |
| `--kmmlu-shots` | 5 | KMMLU few-shot 수 |

### 2. PPL만 단독 평가 — `eval_ppl.py`

```bash
# 양자화 모델 (GPTQModel 포맷)
python eval_ppl.py --model /workspace/LLM-VLM-in-Jetson/Ministral-3-3B-Instruct-2512-BF16_gptq_4bit --quant

# BF16 원본
python eval_ppl.py --model mistralai/Ministral-3-3B-Instruct-2512-BF16
```

### 3. KMMLU만 단독 평가 — `eval_kmmlu.py`

```bash
# 양자화 모델, 45개 과목 전체, 5-shot
python eval_kmmlu.py --model /workspace/LLM-VLM-in-Jetson/Ministral-3-3B-Instruct-2512-BF16_gptq_4bit --quant

# 과목 1개만 빠르게 확인 (문항 20개로 제한)
python eval_kmmlu.py --model mistralai/Ministral-3-3B-Instruct-2512-BF16 \
    --subset Korean-History --limit-per-subject 20 --shots 5
```

### 4. 보고서용 예시 문항 수집

`KMMLU_REPORT.md`의 "5. 실제 질의/추론 예시" 섹션에 넣을 A/B/C/D 확률분포를 뽑을 때 사용. 스크립트 상단의 `BASE_PATH`/`GPTQ_PATH`/`FOEM_PATH` 경로가 하드코딩되어 있으므로 실제 캐시 경로에 맞게 수정 후 실행.

```bash
# BF16 베이스 모델만, 4개 과목(Korean-History/Math/Computer-Science/Marketing)
python collect_base_examples.py

# BF16 → GPTQ 4bit → FOEM 3bit 순으로 로드해 4개 과목(Education/Patent/Law/Information-Technology) 비교
python collect_extra_examples.py
```
출력되는 마크다운 표를 `KMMLU_REPORT.md`에 그대로 붙여넣으면 됨.

### 5. 베이스 모델 포함 3-way 비교 보고서 재생성 — `update_report_with_base.py`

`logs/base_kmmlu.log`(BF16 KMMLU 평가 로그)를 파싱하고, 스크립트에 하드코딩된 기존 GPTQ/FOEM 결과와 합쳐 `KMMLU_REPORT.md`를 재작성한다.

```bash
python update_report_with_base.py
```

사전 조건: `logs/base_kmmlu.log`가 `eval_kmmlu.py --model mistralai/Ministral-3-3B-Instruct-2512-BF16` 실행 결과로 이미 존재해야 함 (`[kmmlu] ...` 형식의 과목별 로그 라인을 정규식으로 파싱).

## 전체 워크플로 예시

```bash
cd FOEM/pseudo

# 1) GPTQ 4-bit, FOEM 3-bit 각각 양자화 (PPL/KMMLU 자동 평가 포함)
python quantize_mistral.py --method gptq --bits 4 --model mistralai/Ministral-3-3B-Instruct-2512-BF16
python quantize_mistral.py --method foem --bits 3 --model mistralai/Ministral-3-3B-Instruct-2512-BF16

# 2) BF16 원본 KMMLU 평가 (로그를 파일로 저장)
python eval_kmmlu.py --model mistralai/Ministral-3-3B-Instruct-2512-BF16 > ../logs/base_kmmlu.log

# 3) 3-way 비교 보고서 생성
python update_report_with_base.py

# 4) 예시 문항 확률분포 추가 수집 (필요 시)
python collect_extra_examples.py
```
