# Ministral-3-3B-Instruct-2512-BF16_foem_3bit

> 생성일: 2026-06-17

## 모델 정보

| 항목 | 값 |
|---|---|
| 원본 모델 | `mistralai/Ministral-3-3B-Instruct-2512-BF16` |
| 양자화 방법 | **FOEM** |
| 비트 | **3-bit** |
| group_size | 128 |
| attn_impl | sdpa |
| offload_to_disk | False |

## 환경

| 항목 | 값 |
|---|---|
| GPU | NVIDIA GeForce RTX 5070 (12 GB) |
| gptqmodel | 7.1.0 |
| transformers | 5.12.1 |
| torch | 2.11.0+cu128 |

## 캘리브레이션

| 항목 | 값 |
|---|---|
| 데이터셋 | allenai/c4 (en/c4-train.00001-of-01024.json.gz) |
| 샘플 수 | 256 |
| batch_size | 4 |

## FOEM 설정

| 파라미터 | 값 |
|---|---|
| alpha | 0.0 |
| beta | 0.2 |

- alpha=0 → 1차 보정(GPTAQ) 비활성화
- beta=0.2 → 직접 오차 피드백 활성화

## PPL 평가 결과

| 항목 | 값 |
|---|---|
| 데이터셋 | WikiText-2 (test) |
| 시퀀스 길이 | 2048 |
| 슬라이딩 윈도우 수 | 147 |
| **PPL** | **10.8654** |
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
| **정확도 (micro)** | **35.58%** |
| 정확도 (macro, 과목 평균) | 35.03% |
| 평가 소요 시간 | 64.9 분 |

<details><summary>과목별 정확도</summary>

| 과목 | 정확도 | 문항 수 |
|---|---:|---:|
| Accounting | 37.00% | 100 |
| Agricultural-Sciences | 27.10% | 1000 |
| Aviation-Engineering-and-Maintenance | 33.60% | 1000 |
| Biology | 32.10% | 1000 |
| Chemical-Engineering | 35.60% | 1000 |
| Chemistry | 38.83% | 600 |
| Civil-Engineering | 30.50% | 1000 |
| Computer-Science | 55.90% | 1000 |
| Construction | 32.30% | 1000 |
| Criminal-Law | 28.50% | 200 |
| Ecology | 31.40% | 1000 |
| Economics | 35.38% | 130 |
| Education | 37.00% | 100 |
| Electrical-Engineering | 29.90% | 1000 |
| Electronics-Engineering | 45.50% | 1000 |
| Energy-Management | 27.20% | 1000 |
| Environmental-Science | 22.20% | 1000 |
| Fashion | 37.00% | 1000 |
| Food-Processing | 35.40% | 1000 |
| Gas-Technology-and-Engineering | 31.30% | 1000 |
| Geomatics | 33.10% | 1000 |
| Health | 37.00% | 100 |
| Industrial-Engineer | 35.70% | 1000 |
| Information-Technology | 53.70% | 1000 |
| Interior-Architecture-and-Design | 40.00% | 1000 |
| Law | 31.00% | 1000 |
| Machine-Design-and-Manufacturing | 35.00% | 1000 |
| Management | 36.80% | 1000 |
| Maritime-Engineering | 40.67% | 600 |
| Marketing | 59.90% | 1000 |
| Materials-Engineering | 35.70% | 1000 |
| Mechanical-Engineering | 28.10% | 1000 |
| Nondestructive-Testing | 38.00% | 1000 |
| Patent | 23.00% | 100 |
| Political-Science-and-Sociology | 37.67% | 300 |
| Psychology | 33.60% | 1000 |
| Public-Safety | 29.50% | 1000 |
| Railway-and-Automotive-Engineering | 29.30% | 1000 |
| Real-Estate | 36.50% | 200 |
| Refrigerating-Machinery | 29.00% | 1000 |
| Social-Welfare | 38.60% | 1000 |
| Taxation | 28.00% | 200 |
| Telecommunications-and-Wireless-Technology | 46.30% | 1000 |
| Korean-History | 33.00% | 100 |
| Math | 23.67% | 300 |

</details>

---

## 양자화 품질

### 전체 통계

| 항목 | 값 |
|---|---|
| 총 양자화 모듈 | 182 (26레이어 × 7모듈) |
| RTN 폴백 | **0 / 182** (0%) |
| 전체 평균 loss | 43.887 |
| 전체 최대 loss | 303.756 |
| 전체 최소 loss | 0.002121 |

### 모듈별 평균 loss

| 모듈 | 평균 loss | 최대 loss | 최소 loss |
|---|---:|---:|---:|
| `self_attn.o_proj` | 0.390 | 2.256 | 0.00212 |
| `mlp.down_proj` | 2.260 | 13.862 | 0.05200 |
| `self_attn.v_proj` | 11.270 | 42.259 | 0.11600 |
| `self_attn.k_proj` | 27.320 | 39.321 | 2.57000 |
| `self_attn.q_proj` | 74.890 | 103.728 | 4.99000 |
| `mlp.up_proj` | 85.610 | 185.965 | 11.47000 |
| `mlp.gate_proj` | 105.470 | 303.756 | 13.05000 |

### loss 상위 5개 모듈

| 레이어 | 모듈 | loss |
|---|---|---:|
| 25 | `mlp.gate_proj` | 303.7563 |
| 24 | `mlp.gate_proj` | 268.1692 |
| 23 | `mlp.gate_proj` | 257.2136 |
| 22 | `mlp.gate_proj` | 218.7183 |
| 21 | `mlp.gate_proj` | 179.8194 |

### 레이어별 평균 loss 추이

| 레이어 | 평균 loss | 시각화 |
|---|---:|---|
|  0 |    4.61 | █ |
|  1 |   10.53 | ██ |
|  2 |   16.64 | ███ |
|  3 |   24.15 | █████ |
|  4 |   27.30 | ██████ |
|  5 |   32.13 | ██████ |
|  6 |   34.82 | ███████ |
|  7 |   40.84 | ████████ |
|  8 |   37.47 | ███████ |
|  9 |   40.44 | ████████ |
| 10 |   37.58 | ███████ |
| 11 |   38.32 | ████████ |
| 12 |   33.24 | ███████ |
| 13 |   37.73 | ████████ |
| 14 |   37.43 | ███████ |
| 15 |   34.96 | ███████ |
| 16 |   42.07 | ████████ |
| 17 |   42.65 | █████████ |
| 18 |   44.42 | █████████ |
| 19 |   51.33 | ██████████ |
| 20 |   58.18 | ████████████ |
| 21 |   64.88 | █████████████ |
| 22 |   75.17 | ███████████████ |
| 23 |   86.82 | █████████████████ |
| 24 |   91.70 | ██████████████████ |
| 25 |   95.72 | ████████████████████ |

---

## 양자화 심층 분석

### 1. 압축률 분석

| 항목 | 값 |
|---|---|
| 원본 형식 | BF16 (16-bit) |
| 양자화 비트 | 3-bit |
| 이론 압축률 (선형 레이어) | 16 / 3 = **5.33×** |
| 원본 선형 레이어 크기 | ~6.0 GB (3.02B params × 2 bytes) |
| 양자화 후 전체 모델 크기 | **2.7 GB** |
| 실질 압축률 (전체 모델) | ~2.8× |

> **이론 대비 실제 차이 이유**: 3-bit으로 양자화되는 것은 선형 레이어(182개)뿐이고,  
> embed_tokens·lm_head·vision tower·layer norms는 BF16으로 유지됨.  
> 또한 group_size=128 오버헤드(scale + zero 파라미터)가 추가됨.

**group_size=128 오버헤드 계산:**
```
scale (fp16): 3.02B params / 128 × 2 bytes ≈  47 MB
zero  (int8): 3.02B params / 128 × 1 byte  ≈  24 MB
합계                                        ≈  71 MB
```

---

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
| `W_fp` | 현재까지 누적된 양자화 오차를 반영한 fp 가중치 |
| `β` | 오차 피드백 강도 (이 실험: 0.2) |

- **① GPTQ 항**: i번째 열을 양자화할 때 발생한 오차를 Hessian 역행렬을 통해 나머지 열에 분산시켜 보상
- **② FOEM β 항**: 이미 쌓인 누적 오차`(W - W_fp)`를 직접 다음 갱신에 반영 → 오차가 전파되지 않고 제자리에서 흡수됨
- 이 실험은 `alpha=0`이므로 1차 보정(GPTAQ)은 비활성, β 항만 동작

---

### 3. loss 지표 해석

양자화 로그에 기록되는 loss의 정체:

```
loss = ||WX - W_q X||²
```

| 기호 | 의미 |
|---|---|
| `W` | 원본 BF16 가중치 행렬 |
| `W_q` | 양자화된 가중치 행렬 |
| `X` | 캘리브레이션 데이터의 입력 활성값 |

**핵심**: 가중치 차이가 아니라 **실제 forward 출력의 차이**를 측정한다.  
같은 오차라도 활성값(X)이 크면 loss가 크게 나온다.

**절대 loss vs. 파라미터당 loss 비교:**

| 모듈 | 절대 loss (최대) | 파라미터 수 | 파라미터당 loss |
|---|---:|---:|---:|
| `mlp.gate_proj` | 303.76 | 28.3M | 1.07 × 10⁻⁵ |
| `mlp.down_proj` | 13.86 | 28.3M | 4.90 × 10⁻⁷ |
| `self_attn.o_proj` | 2.26 | 12.6M | 1.79 × 10⁻⁷ |

→ 절대 loss만 보면 gate_proj가 134× 크지만, 파라미터당으론 60× 차이로 줄어든다.  
→ **절대값 숫자가 크다고 무조건 나쁜 양자화가 아님.**

---

### 4. 모듈별 민감도 원인

**왜 gate_proj / up_proj가 높고, o_proj / down_proj가 낮은가?**

| 모듈 | 입력 → 출력 | 방향 | 절대 loss 평균 |
|---|---|---|---:|
| `mlp.gate_proj` | 3072 → 9216 | **확장** | 105.47 |
| `mlp.up_proj` | 3072 → 9216 | **확장** | 85.61 |
| `self_attn.q_proj` | 3072 → 4096 | 확장 | 74.89 |
| `self_attn.k_proj` | 3072 → 1024 | 축소 | 27.32 |
| `self_attn.v_proj` | 3072 → 1024 | 축소 | 11.27 |
| `mlp.down_proj` | 9216 → 3072 | **축소** | 2.26 |
| `self_attn.o_proj` | 4096 → 3072 | **축소** | 0.39 |

**구조적 이유:**
1. **출력 차원이 클수록** loss 절댓값이 크게 집계됨 (||WX - W_qX||² 의 합산 항이 많아짐)
2. **SwiGLU 구조**: gate_proj 출력이 sigmoid-like 게이트로 작용 → 작은 오차도 비선형적으로 증폭
3. **축소 방향 레이어**는 입력 공간의 중요 성분을 선별적으로 압축하는 역할이라 상대적으로 양자화에 강인

---

### 5. 레이어 깊이 효과

레이어가 깊어질수록 loss가 급증하는 이유:

| 구간 | 레이어 | 평균 loss | 역할 |
|---|---|---:|---|
| 초기 | 0 ~ 7 | 4.6 ~ 40.8 | 기본 어휘·문법 패턴 추출 |
| 중간 | 8 ~ 18 | 37.5 ~ 44.4 | 중간 추상화 (안정 구간) |
| 후기 | 19 ~ 25 | 51.3 ~ 95.7 | 추론·맥락 이해, 고차원 표현 |

**이유:**
- 초기 레이어는 활성값 분포가 단순하고 Hessian이 잘 조건화(well-conditioned)되어 양자화가 쉬움
- 후기 레이어로 갈수록 활성값의 variance가 커지고 Hessian의 고유값 분포가 넓어짐  
  → 3-bit으로 표현해야 할 값의 범위가 넓어져 양자화 오차 급증
- 즉, **모델이 "추론"을 담당하는 레이어일수록 양자화 손실이 크다**
- 이는 3-bit이 4-bit 대비 후기 레이어에서 더 큰 품질 차이를 만드는 원인

---

### 6. 핵심 하이퍼파라미터 의미

**damp_percent = 0.05**

```
H' = H + 0.05 × mean(diag(H)) × I
```
Hessian 역행렬 계산 시 수치 불안정 방지용 정규화.  
- 너무 작으면: Hessian이 거의 특이(singular)할 때 역행렬 계산 폭발 → RTN 폴백 발생  
- 너무 크면: Hessian 왜곡 → 최적 보상 방향 오류 → 오히려 오차 증가  
- 0.05는 업계 표준값 (0/182 RTN 폴백으로 이 실험에서 완벽히 작동)

**group_size = 128**

| group_size | 오버헤드 | 품질 | 용도 |
|---|---|---|---|
| 32 | 높음 | 최상 | 고품질 우선 |
| **128** | **중간** | **양호** | **표준 (이 실험)** |
| 256 | 낮음 | 보통 | 크기 우선 |
| -1 (per-column) | 최대 | 최상 | 연구용 |

128개 가중치마다 독립적인 scale + zero 값을 가지므로, 작은 범위 내에서 최적 스케일링 가능.  
숫자가 작을수록 더 세밀한 조정이 가능하지만 오버헤드가 증가한다.

---

## 사용 방법

```python
from gptqmodel import GPTQModel

model = GPTQModel.from_quantized("/workspace/LLM-VLM-in-Jetson/Ministral-3-3B-Instruct-2512-BF16_foem_3bit")
```

## 파일 구성

| 파일 | 설명 |
|---|---|
| `model.safetensors` | 양자화된 가중치 (2.7 GB) |
| `quantize_config.json` | 양자화 설정 |
| `config.json` | 모델 아키텍처 설정 |
| `tokenizer.json` | 토크나이저 |
| `README.md` | 이 파일 |
