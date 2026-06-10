# llama32-vlm-gptq

Llama 3.2 11B Vision(Instruct) 모델을 **GPTQ 4비트**로 양자화하고, 최종적으로 **Jetson Orin Nano**에 올리는 프로젝트.

> 왜 GPTQ인가 (vs AWQ/SpinQuant): [docs/quantization_method_selection.md](docs/quantization_method_selection.md)

## 단계

- **Phase 1 (현재): 데스크톱 양자화 + 검증**
  - RTX 3080 Ti(12GB)에서 GPTQModel로 레이어 단위 4비트 양자화
  - 양자화 전/후 정확도·VRAM·속도 비교
- **Phase 2 (후속): Jetson Orin Nano 이식**
  - `scripts/jetson/` 참고. 8GB 적재 가능성 / mllama 런타임은 별도 검증 필요

## 비교 설계 (중요)

비교는 **목적이 다른 두 종류**다 — 섞으면 결과 해석이 꼬인다.

1. **순수 양자화 검증** (핵심): `11B fp16` vs `11B 4bit` — **같은 모델**이라 점수
   차이 = 순수 양자화 효과. fp16은 22GB라 Jetson엔 못 올리므로 **정확도는
   데스크톱에서** 측정(정확도는 하드웨어 무관), Jetson엔 4bit만 올려 VRAM·속도 측정.
2. **엣지 배포 참고선**: `SmolVLM2-2.2B`(무양자화) — "큰 모델 양자화 vs 작은 모델
   네이티브"용 **별도 참고선**. 양자화 기준표가 아님(다른 모델이라 양자화 효과를
   분리 못 함). 회의록의 "같은 계열 작은 모델"이 이쪽.

> GPTQModel의 mllama 지원은 **텍스트 레이어 한정** → 캘리브는 텍스트(list[str])로
> 충분하고, 비전 타워는 fp16으로 남는다(Jetson VRAM 계산에 반영).

## 환경

| 항목 | 값 |
|------|-----|
| GPU | NVIDIA RTX 3080 Ti, 12GB |
| CUDA | 12.6 (드라이버 560.94) |
| 시스템 RAM | 64GB (CPU 오프로드용으로 충분) |
| Python | **3.11 권장** (3.13은 gptqmodel/torch 휠 미지원 가능성) |

> 12GB VRAM으로는 11B Vision(fp16 ~22GB)을 통째로 못 올리지만, GPTQModel이
> 레이어 단위로 양자화하고 나머지는 CPU(RAM)로 오프로드하므로 동작 가능.

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

> 드라이버 560.94(CUDA 12.6) 기준 `cu126` 사용. `pip install` 순서를 바꾸면
> torch가 CPU 빌드로 덮여 `torch.cuda.is_available()==False` 가 되니 주의.

## 모델 접근 (필수)

`meta-llama/Llama-3.2-11B-Vision-Instruct`는 **gated 모델**입니다.

1. https://huggingface.co/meta-llama/Llama-3.2-11B-Vision-Instruct 에서 라이선스 승인
2. HF 액세스 토큰 발급 후 `.env`에 설정 (`.env.example` 복사)

## 실행

```powershell
python src/download_model.py     # 1. 원본 모델 다운로드
python src/quantize.py           # 2. GPTQ 4비트 양자화
python src/evaluate.py           # 3. 전/후 비교
```

세부 설정은 `configs/gptq_config.yaml` 에서 조정.
