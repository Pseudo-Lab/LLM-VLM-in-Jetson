# Ministral-3-3B-Instruct-2512-BF16_gptq_4bit

> 생성일: 2026-06-17 09:21

## 모델 정보

| 항목 | 값 |
|---|---|
| 원본 모델 | `mistralai/Ministral-3-3B-Instruct-2512-BF16` |
| 양자화 방법 | **GPTQ** |
| 비트 | **4-bit** |
| group_size | 128 |
| attn_impl | sdpa |
| offload_to_disk | False |

## 환경

| 항목 | 값 |
|---|---|
| GPU | NVIDIA GeForce RTX 5070 |
| gptqmodel | 7.1.0 |
| transformers | 5.12.1 |
| torch | 2.11.0+cu128 |
| 양자화 소요 시간 | 4.7 분 |

## 캘리브레이션

| 항목 | 값 |
|---|---|
| 데이터셋 | allenai/c4 (폴백: wikitext-2) |
| 샘플 수 | 256 |
| batch_size | 1 |

---

## 양자화 심층 분석

### 2. GPTQ 알고리즘 원리

```
ΔW = -e_i ⊗ H⁻¹[i,:]            ← 기본 GPTQ 항
```

- `e_i`: i번째 열 양자화 시 발생한 오차
- `H` (Hessian = XᵀX): 활성값 기반으로 각 가중치의 중요도를 반영
- i번째 열을 양자화할 때 발생한 오차를 Hessian 역행렬을 통해 나머지 열에 분산시켜 보상

---

## PPL 평가 결과

| 항목 | 값 |
|---|---|
| 데이터셋 | WikiText-2 (test) |
| 시퀀스 길이 | 2048 |
| 슬라이딩 윈도우 수 | 147 |
| **PPL** | **8.7238** |
| 평가 소요 시간 | 1.3 분 |

> **PPL (Perplexity)**: 언어 모델이 텍스트를 얼마나 잘 예측하는지 나타내는 지표.
> 낮을수록 좋으며, 양자화 전후 PPL 차이가 클수록 품질 손실이 크다는 의미.
> WikiText-2 는 국제 표준 벤치마크로, 동일 조건에서 서로 다른 양자화 방법 비교 시 사용한다.

---

## KMMLU 평가 결과

| 항목 | 값 |
|---|---|
| 데이터셋 | KMMLU (45개 과목, test) |
| Shot | 5-shot |
| 문항 수 | 35030 |
| **정확도 (micro)** | **44.19%** |
| 정확도 (macro, 과목 평균) | 43.42% |
| 평가 소요 시간 | 62.4 분 |

<details><summary>과목별 정확도</summary>

| 과목 | 정확도 | 문항 수 |
|---|---:|---:|
| Accounting | 37.00% | 100 |
| Agricultural-Sciences | 36.10% | 1000 |
| Aviation-Engineering-and-Maintenance | 41.50% | 1000 |
| Biology | 36.90% | 1000 |
| Chemical-Engineering | 47.30% | 1000 |
| Chemistry | 48.00% | 600 |
| Civil-Engineering | 41.50% | 1000 |
| Computer-Science | 69.20% | 1000 |
| Construction | 34.40% | 1000 |
| Criminal-Law | 33.50% | 200 |
| Ecology | 45.80% | 1000 |
| Economics | 46.92% | 130 |
| Education | 58.00% | 100 |
| Electrical-Engineering | 34.30% | 1000 |
| Electronics-Engineering | 52.90% | 1000 |
| Energy-Management | 33.10% | 1000 |
| Environmental-Science | 30.40% | 1000 |
| Fashion | 44.60% | 1000 |
| Food-Processing | 41.90% | 1000 |
| Gas-Technology-and-Engineering | 36.40% | 1000 |
| Geomatics | 39.80% | 1000 |
| Health | 55.00% | 100 |
| Industrial-Engineer | 40.50% | 1000 |
| Information-Technology | 64.40% | 1000 |
| Interior-Architecture-and-Design | 50.80% | 1000 |
| Law | 42.60% | 1000 |
| Machine-Design-and-Manufacturing | 42.80% | 1000 |
| Management | 49.00% | 1000 |
| Maritime-Engineering | 44.83% | 600 |
| Marketing | 74.90% | 1000 |
| Materials-Engineering | 44.80% | 1000 |
| Mechanical-Engineering | 39.40% | 1000 |
| Nondestructive-Testing | 45.40% | 1000 |
| Patent | 32.00% | 100 |
| Political-Science-and-Sociology | 47.33% | 300 |
| Psychology | 43.20% | 1000 |
| Public-Safety | 38.10% | 1000 |
| Railway-and-Automotive-Engineering | 38.60% | 1000 |
| Real-Estate | 38.50% | 200 |
| Refrigerating-Machinery | 32.70% | 1000 |
| Social-Welfare | 52.60% | 1000 |
| Taxation | 40.00% | 200 |
| Telecommunications-and-Wireless-Technology | 55.50% | 1000 |
| Korean-History | 27.00% | 100 |
| Math | 24.33% | 300 |

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
| 양자화 비트 | 4-bit |
| 이론 압축률 (선형 레이어) | 16 / 4 = **4.00×** |
| 선형 레이어 BF16 추정 크기 | 0.00 GB |
| 실제 모델 파일 크기 | 3.22 GB |
| 실질 압축률 | **0.0×** |
| group_size=128 오버헤드 | ~0 MB (scale+zero 파라미터) |

> 이론 vs 실제 차이: 선형 레이어만 4-bit 양자화되고, embed·lm_head·vision tower·norms는 BF16 유지.

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
→ 4-bit으로 표현해야 할 값의 범위가 넓어져 양자화 오차 급증.
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

model = GPTQModel.from_quantized("/workspace/LLM-VLM-in-Jetson/Ministral-3-3B-Instruct-2512-BF16_gptq_4bit")
```

## 파일 구성

| 파일 | 설명 |
|---|---|
| `model.safetensors` | 양자화된 가중치 (3.2 GB) |
| `quantize_config.json` | 양자화 설정 |
| `config.json` | 모델 아키텍처 설정 |
| `tokenizer.json` | 토크나이저 |
| `README.md` | 이 파일 |
