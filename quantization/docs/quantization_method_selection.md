# 양자화 기법 선정 근거 — 왜 GPTQ인가

> 작성: 2026-06-10 · 대상 모델: `meta-llama/Llama-3.2-11B-Vision-Instruct` (mllama)
> 목표: Jetson Orin Nano (8GB) 온디바이스 배포를 위한 4bit 양자화

## TL;DR

**GPTQ가 알고리즘적으로 최강이라서 고른 게 아니다.**
mllama(Vision) 아키텍처 + Jetson 배포라는 제약 조건에서 **유일하게 성숙한 도구 체인이 GPTQ(gptqmodel)이기 때문**이다. 그리고 우리가 쓰는 W4A16(가중치만 4bit) 세팅에서는 최신 기법과의 품질 격차가 작다.

## 1. 알고리즘 성능만 보면 GPTQ는 1등이 아니다

| 기법 | 연도 | 방식 | 강점 | 약점 |
|------|------|------|------|------|
| **GPTQ** | 2022 | 오차 보상 반올림 (헤시안 기반) | 가장 성숙한 생태계, 도구 최다 | 최신 기법 대비 품질 약간 열세 |
| **AWQ** | 2023 | 활성값 기준 중요 가중치 보호 | W4에서 GPTQ보다 약간 우위 (품질 유지 ~95% vs ~90% 보고) | mllama 지원 도구 없음 |
| **SpinQuant** | 2024 (Meta) | 학습된 회전행렬로 outlier 제거 | **W4A4/W4A8(활성값까지 양자화)에서 압도적** — GPTQ 대비 4점+ 개선. Meta 공식 양자화 Llama 3.2 1B/3B에 실제 사용 | 회전행렬 GPU 학습 필요(파이프라인 무거움), mllama 미지원 |

## 2. 그러나 우리 제약 조건에서는 GPTQ가 정답

### ① SpinQuant의 강점은 우리가 쓰지 않는 영역

- SpinQuant가 빛나는 건 **활성값까지 깎는 W4A4·W4A8** 세팅
- 우리는 **W4A16 (가중치만 4bit, 활성값은 fp16)** → 이 세팅에서는 GPTQ/AWQ/SpinQuant 격차가 크게 줄어듦
- SpinQuant는 회전행렬을 GPU로 학습해야 해서 파이프라인이 훨씬 무겁고, Meta 공식 레포는 텍스트 Llama용 (mllama 미지원)

### ② 도구 생태계 — mllama 지원 여부가 결정적

대상 모델이 일반 Llama가 아닌 **mllama(Vision)** 라는 점이 대부분의 대안을 걸러낸다:

| 도구 | mllama 지원 | 비고 |
|------|:---:|------|
| **gptqmodel** (GPTQ) | ✅ | 텍스트 레이어 양자화 공식 지원 (v1.4.0+), 본 프로젝트에서 검증 완료 |
| AutoAWQ | ❌ | 프로젝트 유지보수 중단 상태 |
| llm-awq (MIT) | ❌ | VLM 지원은 OpenFlamingo/VILA 등에 한정 |
| SpinQuant (Meta) | ❌ | 텍스트 Llama 전용 + 회전 학습 필요 |

→ **"mllama를 4bit로 깎아주는 성숙한 도구"는 사실상 gptqmodel 하나뿐.**
알고리즘이 1~2% 더 좋아도 모델을 아예 못 깎으면 의미가 없다.

### ③ W4 g128에서 품질 격차는 작고, 어차피 자체 측정한다

- group_size=128 가중치 양자화에서 GPTQ vs AWQ 차이는 보통 perplexity 소수점 수준
- 본 프로젝트는 `src/evaluate.py`로 **fp16 vs 4bit 비교표를 직접 생성**하는 설계
  → "양자화 손실이 허용 범위인가"는 추측이 아니라 자체 벤치마크로 정량 검증

## 3. 보고용 한 줄 요약

- "mllama 아키텍처를 지원하는 유일한 성숙 도구 체인(gptqmodel) 기반 GPTQ 선정"
- "W4A16 세팅에서는 최신 기법(SpinQuant) 대비 격차가 작음 — SpinQuant의 이점은 W4A4/W4A8에 집중"
- "양자화 품질은 자체 평가(evaluate.py)로 정량 검증"

## 4. 품질이 부족하게 나올 경우의 폴백 플랜

1. 캘리브레이션 샘플 증량: 256 → 512 (`configs/gptq_config.yaml`의 `num_samples`)
2. `desc_act: true`로 재양자화 (정확도↑ / 추론 속도·호환성↓ 트레이드오프)
3. 그래도 부족하면 그 시점에 AWQ/SpinQuant의 mllama 지원 상황 재조사

## 참고 자료

- [SpinQuant: LLM quantization with learned rotations (arXiv)](https://arxiv.org/pdf/2405.16406)
- [AWQ: Activation-aware Weight Quantization (arXiv)](https://arxiv.org/pdf/2306.00978)
- [llm-awq GitHub (MIT)](https://github.com/mit-han-lab/llm-awq)
- [GPTQModel GitHub (ModelCloud)](https://github.com/ModelCloud/GPTQModel)
- [GPTQ/AWQ/EXL2/llama.cpp 실측 비교 (oobabooga)](https://oobabooga.github.io/blog/posts/gptq-awq-exl2-llamacpp/)
- [Meta 공식 양자화 가이드](https://www.llama.com/docs/how-to-guides/quantization/)
