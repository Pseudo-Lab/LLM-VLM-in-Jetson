# Ministral-3-3B-Instruct-2512-BF16_foem_3bit_kocalib

> 생성일: 2026-07-08 04:39

## 모델 정보

| 항목 | 값 |
|---|---|
| 원본 모델 | `/root/.cache/huggingface/hub/models--mistralai--Ministral-3-3B-Instruct-2512-BF16/snapshots/ecc3ba8b43a45610e709327c049d24b009bfec88` |
| 양자화 방법 | **FOEM** |
| 비트 | **3-bit** |
| group_size | 128 |
| desc_act (act-order) | False |
| attn_impl | eager |
| offload_to_disk | False |

## 환경

| 항목 | 값 |
|---|---|
| GPU | NVIDIA GeForce RTX 5070 |
| gptqmodel | 7.1.0 |
| transformers | 5.12.1 |
| torch | 2.11.0+cu128 |
| 양자화 소요 시간 | 6.0 분 |

## 캘리브레이션

| 항목 | 값 |
|---|---|
| 데이터셋 | wikimedia/wikipedia (20231101.ko, every 10th doc, random 1500자 crop) |
| 샘플 수 | 256 |
| batch_size | 4 |

## FOEM 설정

| 파라미터 | 값 |
|---|---|
| alpha | 0.0 |
| beta | 0.2 |

- alpha=0.0 → 1차 보정(GPTAQ) 비활성화
- beta=0.2 → 직접 오차 피드백 활성화

---

## 양자화 심층 분석

### 2. FOEM 알고리즘 원리

FOEM(First-Order Error Matters, AAAI 2026)은 기본 GPTQ 가중치 갱신식에 오차 피드백 항을 추가한다.

```
ΔW = -e_i ⊗ H⁻¹[i,:]            ← ① 기본 GPTQ 항
     - (W - W_fp) · H⁻² · β     ← ② FOEM 직접 오차 피드백 (β=0.2)
```

| 기호 | 의미 |
|---|---|
| `e_i` | i번째 열 양자화 시 발생한 오차 |
| `H` | 활성값 기반 Hessian 행렬 (XᵀX) |
| `W_fp` | 누적 양자화 오차를 반영한 fp 가중치 |
| `β` | 오차 피드백 강도 (이 실험: 0.2) |

- **① GPTQ 항**: i번째 열 양자화 오차를 Hessian 역행렬로 나머지 열에 분산시켜 보상
- **② FOEM β 항**: 이미 쌓인 누적 오차 `(W - W_fp)`를 다음 갱신에 직접 반영 → 오차가 전파되지 않고 흡수됨

---

## PPL 평가 결과

| 항목 | 값 |
|---|---|
| 데이터셋 | WikiText-2 (test) |
| 시퀀스 길이 | 2048 |
| 슬라이딩 윈도우 수 | 147 |
| **PPL** | **11.1077** |
| 평가 소요 시간 | 1.3 분 |

> **PPL (Perplexity)**: 언어 모델이 텍스트를 얼마나 잘 예측하는지 나타내는 지표.
> 낮을수록 좋으며, 양자화 전후 PPL 차이가 클수록 품질 손실이 크다는 의미.
> WikiText-2 는 국제 표준 벤치마크로, 동일 조건에서 서로 다른 양자화 방법 비교 시 사용한다.

---

> KMMLU 평가 생략 (--skip-kmmlu 옵션 사용)

---

## K-DTCBench 평가 결과

| 항목 | 값 |
|---|---|
| 데이터셋 | NCSOFT/K-DTCBench (document/table/chart, test) |
| Shot | zero-shot |
| 문항 수 | 240 |
| **정확도 (micro=macro)** | **47.50%** |
| 평가 소요 시간 | 1.5 분 |

<details><summary>카테고리별 정확도</summary>

| 카테고리 | 정확도 | 문항 수 |
|---|---:|---:|
| document | 53.75% | 80 |
| table | 47.50% | 80 |
| chart | 41.25% | 80 |

</details>

---

## 양자화 품질

### 전체 통계

| 항목 | 값 |
|---|---|
| 총 양자화 모듈 | 0 |
| RTN 폴백 | **0 / 0** (0%) |
| 전체 평균 loss | 0.0000 |
| 전체 최대 loss | 0.0000 |
| 전체 최소 loss | 0.000000 |

### 모듈별 평균 loss

| 모듈 | 입출력 | 방향 | 평균 loss | 최대 loss | 최소 loss | 파라미터당 loss |
|---|---|---|---:|---:|---:|---:|


> **파라미터당 loss** = 평균 loss ÷ (feat_in × feat_out). 절대 loss가 커도 파라미터 수가 많으면 실제 영향은 작을 수 있음.

### loss 상위 5개 모듈

| 레이어 | 모듈 | loss |
|---|---|---:|


### 레이어별 평균 loss 추이

| 레이어 | 평균 loss | 시각화 |
|---|---:|---|


---

### 1. 압축률 분석

| 항목 | 값 |
|---|---|
| 원본 형식 | BF16 (16-bit) |
| 양자화 비트 | 3-bit |
| 이론 압축률 (선형 레이어) | 16 / 3 = **5.33×** |
| 선형 레이어 BF16 추정 크기 | 0.00 GB |
| 실제 모델 파일 크기 | 2.84 GB |
| 실질 압축률 | **0.0×** |
| group_size=128 오버헤드 | ~0 MB (scale+zero 파라미터) |

> 이론 vs 실제 차이: 선형 레이어만 3-bit 양자화되고, embed·lm_head·vision tower·norms는 BF16 유지.

### 3. loss 지표 해석

양자화 로그의 loss:

```
loss = ||WX - W_q X||²
```

| 기호 | 의미 |
|---|---|
| `W` | 원본 BF16 가중치 행렬 |
| `W_q` | 양자화된 가중치 행렬 |
| `X` | 캘리브레이션 입력 활성값 |

**핵심**: 가중치 자체의 차이가 아닌 **실제 forward 출력의 차이**를 측정.
활성값(X)이 크면 loss도 크게 나오므로 절대값보다 파라미터당 loss가 더 공정한 비교 지표.

### 4. 모듈별 민감도 원인

- **확장 방향** (gate_proj, up_proj, q_proj): 출력 차원이 커서 loss 절댓값이 크게 집계됨.
  SwiGLU 구조에서 gate_proj는 sigmoid-like 게이트로 작용 → 작은 오차도 비선형적으로 증폭 가능.
- **축소 방향** (down_proj, o_proj, k_proj, v_proj): 입력 공간에서 중요한 성분을 선택적으로 압축.
  상대적으로 양자화에 강인하며 파라미터당 loss가 낮음.

### 5. 레이어 깊이 효과

| 구간 | 평균 loss | 역할 |
|---|---:|---|
| 초기 (0~0) | 0.0 | 기본 어휘·문법 패턴 추출 |
| 중간 (1~1) | 0.0 | 중간 추상화 (안정 구간) |
| 후기 (2~0) | 0.0 | 추론·맥락 이해, 고차원 표현 |


후기 레이어일수록 활성값 variance가 크고 Hessian 고유값 분포가 넓어짐
→ 3-bit으로 표현해야 할 값의 범위가 넓어져 양자화 오차 급증.
→ **모델이 "추론"을 담당하는 레이어일수록 양자화 손실이 크다.**

### 6. 핵심 하이퍼파라미터 의미

**damp_percent = 0.05**
```
H' = H + 0.05 × mean(diag(H)) × I
```
Hessian 역행렬 계산 수치 안정화용 정규화. 이 실험에서 RTN 폴백 0건으로 완벽히 작동.

**group_size = 128**

| group_size | 오버헤드 | 품질 | 용도 |
|---|---|---|---|
| 32 | 높음 | 최상 | 고품질 우선 |
| **128** | **중간** | **양호** | **이 실험 (표준값)** |
| 256 | 낮음 | 보통 | 크기 우선 |

---

## 사용 방법

```python
from gptqmodel import GPTQModel

model = GPTQModel.from_quantized("/workspace/LLM-VLM-in-Jetson/Ministral-3-3B-Instruct-2512-BF16_foem_3bit_kocalib")
```

## 파일 구성

| 파일 | 설명 |
|---|---|
| `model.safetensors` | 양자화된 가중치 (2.8 GB) |
| `quantize_config.json` | 양자화 설정 |
| `config.json` | 모델 아키텍처 설정 |
| `tokenizer.json` | 토크나이저 |
| `README.md` | 이 파일 |
