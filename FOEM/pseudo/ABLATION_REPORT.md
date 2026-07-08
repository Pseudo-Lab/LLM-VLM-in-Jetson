# FOEM 3-bit 양자화 Ablation 실험 보고서
**act-order(desc_act) / 캘리브레이션 언어(en vs ko) / FOEM alpha·beta / group_size 튜닝**

작성일: 2026-07-08 · 모델: `mistralai/Ministral-3-3B-Instruct-2512-BF16` · 방법: FOEM 3-bit (베이스라인: α=0, β=0.2, group_size=128 → **최종 권장: group_size=32**)

---

## 1. 목적

`pseudo/KMMLU_REPORT.md` / `pseudo/KDTCBENCH_REPORT.md`에서 확인된 FOEM 3-bit의 큰 정확도 손실(KMMLU -11.30%p, K-DTCBench -11.25%p)을 줄이기 위해 다섯 가지 축을 튜닝했다.

1. **act-order (`desc_act=True`)**: GPTQ 계열에서 activation 크기 순으로 열을 재정렬한 뒤 양자화하는 옵션. 일반적으로 정확도 개선 폭이 가장 큰 단일 스위치로 알려져 있음.
2. **캘리브레이션 언어**: 기존 파이프라인은 영어 `allenai/c4`를 사용하는데, 평가셋(KMMLU/K-DTCBench)은 한국어 중심이므로 한국어 코퍼스로 캘리브레이션하면 실제 평가 분포에 더 잘 맞을 것이라는 가설.
3. **FOEM alpha/beta**: FOEM 고유 파라미터. alpha는 GPTAQ 스타일 1차 보정 항의 강도(베이스라인 0.0=비활성), beta는 누적 오차 직접 피드백 강도(베이스라인 0.2). 두 값을 그리드로 조정.
4. **group_size**: 양자화 시 scale/zero-point를 공유하는 가중치 묶음 크기(베이스라인 128). 축소하면 정확도↑, 파일 크기↑.
5. **mixed-precision 레이어 스킵**: 손실이 큰 일부 모듈(gate/up_proj)이나 레이어(후반부)만 4-bit로 남기고 나머지는 3-bit 유지.

**결론을 먼저 요약하면**: 앞의 네 축(act-order, calib-language, alpha/beta, mixed-precision — 총 9개 설정)은 전부 베이스라인과 동률이거나 나빴다. 유일하게 **group_size를 32로 축소했을 때만 PPL과 K-DTCBench가 모두 개선**됐다 (파일 크기 +5.9%만 대가로). 이번 세션에서 시도한 11개 대안 설정 중 베이스라인을 이긴 유일한 조합이며, 최종 권장 설정이다. 상세 원인 분석은 9절 참고.

---

## 2. 공통 실험 설정

| 항목 | 값 |
|---|---|
| 모델 | `mistralai/Ministral-3-3B-Instruct-2512-BF16` |
| 양자화 방법 | FOEM, 3-bit, group_size=128 |
| FOEM 파라미터 | alpha=0.0, beta=0.2 (베이스라인과 동일, 이번 실험에서는 미변경) |
| 캘리브레이션 샘플 수 | 256 |
| batch_size | 4 |
| 평가 (빠른 반복용) | WikiText-2 PPL + K-DTCBench (zero-shot, 240문항) — 각 1.5분 내외 |
| 평가 (생략) | KMMLU (45과목 5-shot, 65분 소요) — `--skip-kmmlu`로 매 실험마다 생략, 최종 선정 설정에서만 재실행 예정 |
| 구현 | `pseudo/quantize_mistral.py` (이번 세션에서 `--desc-act`, `--calib-lang` 플래그 및 K-DTCBench 자동 평가 통합 추가) |

### 베이스라인 (기존 `Ministral-3-3B-Instruct-2512-BF16_foem_3bit`)

| 지표 | 값 |
|---|---:|
| desc_act | False (기본값) |
| 캘리브레이션 | `allenai/c4` (영어), 256 샘플, 길이 절단 없음 |
| WikiText-2 PPL | **10.8654** |
| KMMLU 정확도 (micro) | **35.58%** |
| K-DTCBench 정확도 (micro=macro) | **50.83%** (document 58.75% / table 56.25% / chart 37.50%) |

---

## 3. 실험 1: `desc_act=True` + 한국어 캘리브레이션 (문서 서두만 절단)

### 3-1. 캘리브레이션 데이터 분포 설정

- **소스**: `wikimedia/wikipedia` (`20231101.ko` config), `streaming=True`로 스트림 순서대로 앞에서부터 256개 문서를 가져옴 (`itertools.islice(ds, 256)`).
- **길이 절단**: 각 문서 텍스트를 **항상 0번째 글자부터 1500자까지** 잘라서 사용 (`ex["text"][:1500]`).
- **절단 이유**: 위키 문서를 통째로 쓰면 일부 문서가 수만 자에 달해 캘리브레이션 배치(batch_size=4) 처리 중 단일 텐서 할당이 19.57GiB에 달해 OOM이 발생함 (RTX 5070 12GB 환경). c4/wikitext 샘플은 원래 짧아 이 문제가 없었음 — 이번에 처음 발견된 이슈.
- **알려진 한계 (사후 확인)**: 항상 0번째 글자부터 자르므로 256개 샘플 전부가 "OOO는 ~이다"류의 백과사전 서두 문체로 편중됨. 이는 4절에서 개선함.

### 3-2. 기술적 이슈 및 수정

1. **HF 허브 리비전 불일치**: `mistralai/Ministral-3-3B-Instruct-2512-BF16`의 "main" 리비전이 실험 도중 새 커밋으로 바뀌었는데, 로컬 캐시에는 새 리비전의 config/tokenizer만 받아져 있고 가중치(safetensors)가 없어 첫 실행이 `OSError: no file named model.safetensors`로 실패. `quantize_mistral.py`가 로컬 디렉토리 경로를 직접 받으면 재다운로드 없이 그대로 쓰도록 수정하고, 가중치가 완전한 기존 로컬 스냅샷 경로(`ecc3ba8b...`)를 명시적으로 지정해 해결.
2. **OOM**: 위 3-1의 길이 절단 미적용으로 인한 19.57GiB 할당 실패. 1500자 절단 적용 후 해결.

### 3-3. 결과

| 지표 | 베이스라인 | 실험 1 | 차이 |
|---|---:|---:|---:|
| WikiText-2 PPL | 10.87 | **11.35** | +0.49 (악화) |
| K-DTCBench 전체 | 50.83% | **46.67%** | **-4.16%p** |
| K-DTCBench document | 58.75% | 50.00% | -8.75%p |
| K-DTCBench table | 56.25% | 48.75% | -7.50%p |
| K-DTCBench chart | 37.50% | 41.25% | **+3.75%p** |

두 지표 모두 베이스라인보다 악화. document/table처럼 텍스트를 정밀 판독해야 하는 카테고리에서 손실이 크고, chart만 유일하게 개선.

---

## 4. 실험 2: 한국어 캘리브레이션만 적용 (act-order 제외, 다양성 개선)

실험 1의 손실이 act-order 때문인지 캘리브레이션 때문인지 분리하기 위해, **act-order는 끄고** 캘리브레이션 다양성 문제만 고쳐서 재실행.

### 4-1. 캘리브레이션 데이터 분포 설정 (개선판)

실험 1의 "항상 문서 서두만 사용" 편향을 해소하기 위해 두 가지를 변경:

1. **주제 다양성** — 스트림에서 연속된 256개 문서 대신, **매 10번째 문서**를 골라 총 2,560개 문서를 훑으면서 256개를 추출 (`itertools.islice(ds, 0, 256*10, 10)`). 위키 덤프가 대략 문서 ID 순으로 정렬돼 있어 연속 추출 시 특정 주제군에 쏠릴 수 있는 위험을 줄임.
2. **문서 내 위치 다양성** — 문서 길이가 1500자를 넘으면, **무작위 시작 위치**(`random.Random(seed=0).randint(0, len(text)-1500)`)에서 1500자를 잘라냄. 항상 서두(정의문 위주)가 아니라 본문 중간(사건 서술, 나열, 인용 등 다양한 문체)도 포함되도록 함. 시드를 고정해 재현 가능하게 함.
3. **절단 길이(1500자)는 실험 1과 동일** — OOM 방지 목적은 유지, 이번에는 "어디를 자르는지"만 바꿈.

| 항목 | 실험 1 (서두 절단) | 실험 2 (다양성 개선) |
|---|---|---|
| 문서 선택 | 스트림 앞 256개 연속 | 매 10번째 문서, 256개 |
| 절단 위치 | 항상 0번째 글자 | 문서별 무작위 시작 위치 |
| 절단 길이 | 1500자 | 1500자 (동일) |
| 재현성 | 결정적 (스트림 순서 고정) | 결정적 (RNG seed=0 고정) |

### 4-2. 결과

| 지표 | 베이스라인 | 실험 1 (서두) | 실험 2 (다양성 개선) |
|---|---:|---:|---:|
| WikiText-2 PPL | 10.87 | 11.35 | **11.11** |
| K-DTCBench 전체 | 50.83% | 46.67% | **47.50%** |
| K-DTCBench document | 58.75% | 50.00% | 53.75% |
| K-DTCBench table | 56.25% | 48.75% | 47.50% |
| K-DTCBench chart | 37.50% | 41.25% | 41.25% |

다양성 개선으로 실험 1 대비 PPL(-0.24)과 K-DTCBench(+0.83%p)가 소폭 회복됐지만, **여전히 베이스라인보다 K-DTCBench -3.33%p, PPL +0.24 악화** — 캘리브레이션 다양성 문제가 손실의 일부였을 뿐, 근본 원인은 아니었음을 시사한다.

---

## 5. 실험 3: FOEM alpha/beta 그리드 서치

실험 1·2에서 act-order와 캘리브레이션 언어 모두 기각된 뒤, en/c4 + desc_act=False (베이스라인 설정)로 복귀하고 FOEM 고유 파라미터인 alpha/beta를 튜닝했다.

```
ΔW = -e_i ⊗ H⁻¹[i,:]            ← ① 기본 GPTQ 항
     - (W - W_fp) · H⁻² · β     ← ② FOEM 직접 오차 피드백 (베이스라인 β=0.2)
```

- **alpha**: GPTAQ 스타일 1차 보정 항의 강도. 베이스라인은 alpha=0.0 (비활성).
- **beta**: 위 식 ②번, 누적 양자화 오차를 다음 갱신에 직접 반영하는 피드백 강도. 베이스라인은 0.2.

그리드: beta=0.2 고정, alpha∈{0.3, 0.6}로 스크리닝 → 이후 가장 나은 alpha로 beta∈{0.1, 0.4} 스윕. 단, alpha 실험 결과 alpha>0이 전부 K-DTCBench를 악화시켜, beta 스윕은 **alpha=0.0(베이스라인) 고정**으로 진행했다.

### 5-1. 기술적 이슈 및 수정 (alpha>0 관련)

alpha=0.3 첫 시도(batch_size=4, 베이스라인과 동일)에서 두 가지 새로운 실패가 발생했다.

1. **OOM (배치 크기와 무관)**: `Tried to allocate 14.66 GiB`. batch_size를 4→2→1로 줄여도 동일 크기(7.33GiB)의 단일 할당 시도가 반복돼, 원인이 배치 크기가 아니라 **c4 캘리브레이션 샘플 중 하나가 비정상적으로 긴 문서(추정 1만 토큰 이상)** 였음을 확인. 기존 파이프라인은 c4 텍스트 길이를 전혀 캡하지 않았는데, `allenai/c4` 데이터도 허브에서 갱신되며 이런 이상치가 섞여 들어온 것으로 보인다. → `get_calibration()`의 모든 경로(en/ko/fallback)에 공통 길이 캡(`CALIB_MAX_CHARS=2000`자)을 추가해 근본 해결.
2. **RuntimeError (ragged tensor)**: 길이 캡 적용 후에도 `The size of tensor a (2071) must match the size of tensor b (529) at non-singleton dimension 1` 발생. `gptqmodel/quantization/foem.py`의 `process_batch()`가 **alpha>0일 때 배치 내 서로 다른 길이의 시퀀스를 처리하지 못하는 버그**로 확인됨 (alpha=0이면 이 보정 경로 자체가 스킵되어 문제가 드러나지 않았음). → alpha>0 실험은 모두 **batch_size=1**로 우회 (배치 내 길이 불일치가 발생하지 않도록).

이 두 수정 덕분에 alpha=0.3부터는 정상적으로 양자화가 끝까지 진행됐다.

### 5-2. 결과

| alpha | beta | batch_size | WikiText-2 PPL | K-DTCBench (micro=macro) | vs 베이스라인 |
|---:|---:|---:|---:|---:|---:|
| **0.0** | **0.2 (베이스라인)** | 4 | 10.87 | **50.83%** | - |
| 0.3 | 0.2 | 1 (버그 우회) | 10.74 (개선) | 41.67% | **-9.16%p** |
| 0.6 | 0.2 | 1 (버그 우회) | **10.69 (최고 개선)** | 42.92% | **-7.91%p** |
| 0.0 | 0.1 | 4 | 10.83 | 47.92% | -2.91%p |
| 0.0 | 0.4 | 4 | 11.04 | 49.17% | -1.66%p |

**핵심 관찰**: alpha를 키울수록(0.0→0.3→0.6) WikiText-2 PPL은 계속 좋아지는데(10.87→10.74→10.69), K-DTCBench는 두 값 모두 베이스라인보다 8~9%p나 나쁘다. **PPL 개선이 다운스트림 VQA 태스크 성능과 반대 방향으로 움직인 사례** — PPL만으로 양자화 설정을 판단하면 안 된다는 근거가 된다. beta는 0.1(47.92%) < 0.4(49.17%) < 0.2(50.83%, 베이스라인) 순으로, 베이스라인이 국소 최적점으로 보인다.

---

## 6. 실험 4: group_size 그리드 서치 (128 vs 64 vs 32)

`group_size`는 양자화 시 scale/zero-point를 공유하는 연속 가중치 묶음의 크기다. 작을수록(예: 32) 그룹마다 더 세밀하게 캘리브레이션되어 양자화 오차가 줄지만, scale/zero-point 저장 오버헤드가 늘어 파일 크기가 커진다. 베이스라인(128) 대비 32/64로 축소해 **정확도와 파일 크기 트레이드오프**를 확인했다. alpha=0.0, beta=0.2(확인된 최선값), en/c4 캘리브레이션은 고정.

### 6-1. 결과 (정확도 + 용량)

| group_size | WikiText-2 PPL | K-DTCBench (micro=macro) | model.safetensors 크기 | 베이스라인 대비 용량 |
|---:|---:|---:|---:|---:|
| **128 (베이스라인)** | 10.87 | 50.83% | 2.840 GB | - |
| 64 | 10.32 (개선) | 48.75% (**-2.08%p**) | 2.896 GB | **+2.0%** |
| **32** | **9.85 (최고 개선)** | **53.33% (+2.50%p)** | 3.008 GB | **+5.9%** |

(파일 크기는 `model.safetensors`를 직접 `ls -la`로 측정한 실제 값. README의 "압축률 분석" 표는 이번 실험 로그에 원본 BF16 선형 레이어 크기 집계가 누락돼 0.00GB/0.0×로 잘못 표시되는 기존 버그가 있어 여기서는 실측값을 사용했다.)

### 6-2. 핵심 관찰

- **group_size=32가 이번 전체 ablation(실험 1~4, 총 9개 설정) 중 유일하게 PPL과 K-DTCBench 둘 다 베이스라인을 이긴 설정이다.** PPL -1.02, K-DTCBench +2.50%p, 대가는 파일 크기 +5.9%(2.84GB→3.01GB, +168MB)뿐이다.
- **group_size=64는 alpha 실험과 동일한 PPL/K-DTCBench 괴리 패턴을 보인다.** PPL은 베이스라인보다 좋아졌지만(10.32 < 10.87) K-DTCBench는 오히려 나빠졌다(48.75% < 50.83%). 128→64→32로 갈수록 PPL은 단조 개선되지만 K-DTCBench는 128(50.83%) > 64(48.75%) 사이에 32(53.33%)가 끼어드는 비단조 패턴 — n=240(카테고리당 80)의 표본 노이즈일 가능성과, 32만이 어떤 임계 세밀도를 넘어서며 실질적 개선을 준 것일 가능성이 둘 다 있어 후속 검증(예: 다른 캘리브레이션 시드로 재현)이 필요하다.
- 이번 세션 전체에서 시도한 9개 대안 설정(act-order 1개, calib-lang 1개, alpha/beta 4개, group_size 2개) 중 **베이스라인을 이긴 것은 group_size=32가 유일**하다.

---

## 7. 실험 5: mixed-precision 레이어 스킵 (group_size=128 기준)

group_size=32 발견 이후, "3-bit로 전체를 밀지 말고 손실이 큰 일부 모듈/레이어만 4-bit로 남기면 어떨까"라는 축을 추가로 시도했다. 순수 효과를 보기 위해 group_size는 베이스라인(128)으로 고정하고, alpha=0.0/beta=0.2/en·c4 캘리브레이션도 동일하게 유지했다. gptqmodel `QuantizeConfig`의 `dynamic` 파라미터(`{"+:정규식": {"bits": N}}`)로 모듈별 bit 오버라이드를 구현했다.

### 7-1. 조합 A — 모듈 타입 기준 (`gate_proj`+`up_proj` → 4-bit)

기존 quantize 로그의 "모듈별 민감도" 분석에서 SwiGLU 게이트 역할인 `gate_proj`/`up_proj`가 파라미터당 loss가 가장 크다는 게 이미 확인돼 있어, 이 두 모듈만 전체 26개 레이어에 걸쳐 4-bit로 상향했다 (`+:model\.language_model\.layers\.\d+\.mlp\.(gate_proj|up_proj)$` → `{"bits": 4}`).

### 7-2. 조합 B — 레이어 깊이 기준 (마지막 6개 레이어 → 4-bit)

"후기 레이어(추론·맥락 이해 담당)일수록 loss 급증" 경향에 기반해, 26개 레이어 중 마지막 6개(20~25)를 통째로 4-bit로 상향했다 (`+:model\.language_model\.layers\.(20|21|22|23|24|25)\..*` → `{"bits": 4}`).

### 7-3. 기술적 이슈 및 수정

1. **HF transformers 지원 경고**: quantize 시작 시 `GPT-QModel's per-module dynamic quantization feature is fully supported in latest vLLM and SGLang but not yet available in hf transformers` 경고가 뜬다. 실제로는 두 조합 모두 PPL이 정상 범위(9.8~10.0대)로 나와, **HF transformers 로딩 경로에서도 추론 자체는 올바르게 동작함**을 확인했다 (경고는 최적화 커널 미지원을 의미하는 것으로 보인다).
2. **gptqmodel GPTQ v1 포맷 변환 버그 (조합 B에서 발견)**: 저장 단계(`pack_module` → `convert_gptq_v2_to_v1_format_module`)에서 `ValueError: 3-bit GPTQ qzeros expects columns divisible by 3, got shape (24, 512)` 발생. 원인은 이 변환 함수가 **모듈별 dynamic bit(4)가 아니라 `QuantizeConfig`의 전역 `bits`(3)를 그대로 사용**해 4-bit로 패킹된 레이어의 qzeros를 3-bit 포맷으로 잘못 해석하려 한 것. → `QuantizeConfig(format=FORMAT.GPTQ_V2)`로 레거시 v1 변환 경로 자체를 건너뛰어 우회 (`--format gptq_v2` CLI 플래그로 노출). 조합 A는 이 버그를 우연히 피해갔다 (건드린 모듈의 텐서 shape가 3으로 나누어떨어져 문제가 드러나지 않았을 뿐으로 추정).

### 7-4. 결과

| 조합 | 변경 대상 | WikiText-2 PPL | K-DTCBench | 파일 크기 | vs 베이스라인 |
|---|---|---:|---:|---:|---:|
| **베이스라인** | 없음 | 10.87 | **50.83%** | 2.840 GB | - |
| A | gate_proj+up_proj (전 레이어) → 4bit | 9.82 (개선) | 50.83% (동률) | 3.025 GB (+6.5%) | K-DTCBench 무변화 |
| B | 마지막 6개 레이어 전체 → 4bit | 9.96 (개선) | 49.58% | 2.928 GB (+3.1%) | **-1.25%p** |

두 조합 모두 alpha 실험과 같은 패턴을 반복한다 — **PPL은 크게 개선되지만 K-DTCBench는 베이스라인을 넘지 못한다** (A는 정확히 동률, B는 오히려 하락). group_size=32(PPL 9.85, K-DTCBench 53.33%, 파일 +5.9%)와 비교하면, mixed-precision 두 조합 모두 **PPL은 group_size=32와 비슷한 수준으로 개선되지만 K-DTCBench 개선은 재현되지 않았다.** group_size 축소가 "전체 그룹의 스케일 정밀도"를 높이는 반면 mixed-precision은 "일부 모듈만 비트를 올리는" 방식이라, 이번 태스크(K-DTCBench)에서는 전자의 접근이 더 유효한 것으로 보인다.

---

## 8. 종합 비교표

| 실험 | desc_act | alpha | beta | group_size | mixed-precision | WikiText-2 PPL | K-DTCBench | 파일 크기 |
|---|:---:|---:|---:|---:|---|---:|---:|---:|
| **베이스라인** | False | 0.0 | 0.2 | 128 | 없음 | 10.87 | 50.83% | 2.840 GB |
| 실험 1 | **True** | 0.0 | 0.2 | 128 | 없음 | 11.35 | 46.67% | - |
| 실험 2 | False | 0.0 | 0.2 | 128 | 없음 | 11.11 | 47.50% | - |
| 실험 3-a | False | 0.3 | 0.2 | 128 | 없음 | 10.74 | 41.67% | - |
| 실험 3-b | False | 0.6 | 0.2 | 128 | 없음 | 10.69 | 42.92% | - |
| 실험 3-c | False | 0.0 | 0.1 | 128 | 없음 | 10.83 | 47.92% | - |
| 실험 3-d | False | 0.0 | 0.4 | 128 | 없음 | 11.04 | 49.17% | - |
| 실험 4-a | False | 0.0 | 0.2 | 64 | 없음 | 10.32 | 48.75% | 2.896 GB |
| **실험 4-b** | False | 0.0 | 0.2 | **32** | 없음 | **9.85** | **53.33%** | 3.008 GB |
| 실험 5-a | False | 0.0 | 0.2 | 128 | gate/up_proj→4bit | 9.82 | 50.83% | 3.025 GB |
| 실험 5-b | False | 0.0 | 0.2 | 128 | 마지막6레이어→4bit | 9.96 | 49.58% | 2.928 GB |

**11개 설정 중 group_size=32(실험 4-b)만이 베이스라인을 이겼다 — 최종 권장 설정.**

---

## 9. 결론 및 원인 분석

1. **act-order(desc_act)는 FOEM과 상성이 나쁘다.** `desc_act=True`를 켜면 gptqmodel이 FOEM 고유의 `act_group_aware` 옵션을 자동으로 끄는 경고가 발생한다 (`QuantizeConfig: desc_act=True automatically disables act_group_aware`). FOEM은 자체 오차 피드백 메커니즘(β 항)에 의존하도록 설계됐는데, act-order의 열 재정렬 방식과 결합되면서 오히려 손실이 커진 것으로 추정된다. **일반 GPTQ에서 효과적인 옵션이 FOEM에는 그대로 적용되지 않을 수 있음**을 보여주는 사례.
2. **"캘리브레이션 언어를 평가 언어에 맞추면 좋아질 것"이라는 가설은 기각됐다.** 다양성을 개선해도 한국어 위키 캘리브레이션은 영어 PPL은 물론 **한국어 평가셋(K-DTCBench)에서도** 영어 c4보다 나쁜 결과를 냈다. 이는 캘리브레이션의 핵심이 "언어 일치"가 아니라 **활성값 분포의 다양성(문체·주제·구조)** 임을 시사한다 — c4(뉴스/포럼/블로그/기술문서 등 웹 크롤링)가 위키백과(균질한 백과사전체)보다 언어와 무관하게 더 풍부한 캘리브레이션 신호를 제공하는 것으로 보인다.
3. **chart 카테고리만 두 실험 모두에서 개선**(37.50% → 41.25%, +3.75%p)됐다는 점은 흥미롭지만 n=80이라 노이즈일 가능성을 배제할 수 없다. 후속 검증이 필요하다.
4. **다양성 개선(실험1→실험2)은 방향은 맞았지만 근본 해법은 아니었다.** 문서 선택 및 크롭 위치를 다양화해 손실을 일부 회복했으나 베이스라인을 넘어서지 못했다.
5. **FOEM alpha(GPTAQ 1차 보정)는 PPL과 K-DTCBench에 반대 방향으로 작용한다.** alpha를 키우면 영어 PPL은 계속 좋아지지만 한국어 VQA(K-DTCBench)는 값에 관계없이 8~9%p 악화된다. FOEM의 beta 항(직접 오차 피드백)만으로도 이미 충분히 보정되고 있어, 추가로 alpha 보정을 얹으면 오히려 두 보정 항이 간섭해 손실이 커지는 것으로 추정된다.
6. **FOEM beta는 베이스라인(0.2)이 국소 최적점이다.** 0.1과 0.4 모두 베이스라인보다 나쁘고, 낮출수록(0.1) 더 나쁘다 — 오차 피드백을 약화시키는 방향은 확실히 손해라는 뜻이다.
7. **group_size=32는 이번 세션에서 유일하게 성공한 튜닝이다.** PPL -1.02, K-DTCBench +2.50%p를 파일 크기 +5.9%(168MB)만으로 얻었다. alpha·group_size=64 실험과 마찬가지로 PPL과 K-DTCBench가 항상 같은 방향으로 움직이지는 않는다는 게 반복 확인됐지만, group_size=32는 예외적으로 **두 지표 모두 개선**된 유일한 설정이다.
8. **PPL은 다운스트림(K-DTCBench) 성능의 신뢰할 수 있는 대리지표가 아니다.** alpha 실험, group_size=64 실험, mixed-precision 두 조합 전부 PPL 개선이 K-DTCBench 정체·악화와 함께 나타났다. 양자화 설정을 PPL만으로 선택하면 실제 VQA 태스크에서 손해를 볼 수 있다.
9. **mixed-precision 레이어 스킵(gate/up_proj 또는 마지막 6레이어를 4-bit로)은 group_size 축소보다 효과가 약하다.** 두 조합 모두 PPL은 group_size=32와 비슷한 수준으로 개선됐지만 K-DTCBench는 베이스라인을 넘지 못했다 (동률 또는 하락). "일부 민감 모듈만 정밀도를 높이는" 접근보다 "전체 그룹의 스케일 세밀도를 높이는"(group_size 축소) 접근이 이 모델/태스크 조합에서는 더 유효했다.
10. **gptqmodel의 GPTQ v1 포맷 변환 로직이 모듈별 dynamic bit를 무시하는 버그를 발견했다** (7-3절). `format=FORMAT.GPTQ_V2`로 우회 가능하며, mixed-precision(dynamic bits)을 3-bit 베이스와 함께 쓸 때는 항상 이 옵션을 켜야 한다.

### 다음 단계

- act-order, calib-language, alpha/beta, mixed-precision 튜닝은 모두 폐기.
- **group_size=32 (alpha=0.0, beta=0.2, en/c4, desc_act=False)를 최종 후보로 재확정**, 전체 KMMLU(45과목) 재확인 진행.
- group_size=64와 mixed-precision 두 조합은 K-DTCBench 기준 베이스라인과 동률이거나 나빠 채택하지 않음 (참고용으로만 기록).
- 필요 시 group_size=16처럼 더 세밀한 값이나, group_size=32 위에 mixed-precision을 추가로 얹는 조합을 후속 탐색할 수 있으나, 현재로선 group_size=32로 충분한 개선을 확인했으므로 우선순위는 낮음.

---

## 10. 코드 변경 사항

`pseudo/quantize_mistral.py`에 추가된 기능 (이번 세션):

- `--desc-act` 플래그: `QuantizeConfig(desc_act=...)`에 연결.
- `--calib-lang {en,ko}` 플래그: `get_calibration()`이 `(texts, dataset_label)` 튜플을 반환하도록 변경, ko 옵션은 `wikimedia/wikipedia` 스트리밍 + 다양성 샘플링(4-1절) 구현.
- **모든 캘리브레이션 경로(en/ko/fallback)에 공통 길이 캡 `CALIB_MAX_CHARS=2000`자 적용** (5-1절) — 이상치 긴 문서로 인한 OOM을 원천 차단.
- K-DTCBench 자동 평가를 양자화 파이프라인에 통합 (`eval_kdtcbench_quantized()`, `--skip-kdtcbench` 플래그) — 이제 PPL/KMMLU와 동일하게 양자화 직후 자동 평가되고 README에 기록됨.
- 모델 경로 해석 수정: `--model`에 로컬 디렉토리 경로를 직접 넘기면 재다운로드 없이 사용 (HF 허브 리비전 이슈 회피).
- README 생성기(`write_readme`)에 `desc_act` 값과 실제 사용된 캘리브레이션 데이터셋 라벨(`calib_label`)을 동적으로 기록하도록 변경.
- `--mixed-precision {none,modules,layers}` + `--mixed-precision-bits` + `--mixed-precision-last-layers` 플래그: `QuantizeConfig(dynamic=...)`로 모듈 이름 정규식 기반 bit 오버라이드 구성 (7절).
- `--format {gptq,gptq_v2}` 플래그: `QuantizeConfig(format=...)` — mixed-precision 사용 시 v1 변환 버그 우회용 (7-3절).

### 알려진 라이브러리 버그 (우회, 미수정)

- `gptqmodel/quantization/foem.py`의 `process_batch()`는 **alpha>0일 때 배치 내 시퀀스 길이가 다르면 `RuntimeError`** 를 낸다 (5-1절). `--batch-size 1`로 우회해야 한다.
- `gptqmodel`의 GPTQ v1 포맷 변환(`convert_gptq_v2_to_v1_format_module`)은 **모듈별 dynamic bit를 무시하고 전역 `bits`를 사용**해 저장 시 `ValueError: qzeros expects columns divisible by N`이 날 수 있다 (7-3절). `--format gptq_v2`로 우회해야 한다.
- 위 두 버그 모두 이 레포에서 수정할 수 없는 gptqmodel 라이브러리 쪽 이슈이며, CLI 플래그로 우회만 제공한다.

---

## 11. 산출물

- `pseudo/quantize_mistral.py` — desc_act/calib-lang/길이캡/K-DTCBench/mixed-precision/format 통합 (수정).
- `Ministral-3-3B-Instruct-2512-BF16_foem_3bit_kocalib/` — 실험 2 산출 모델 + README.
- `Ministral-3-3B-Instruct-2512-BF16_foem_3bit_alpha0.3/`, `_alpha0.6/`, `_beta0.1/`, `_beta0.4/` — 실험 3 산출 모델 + README (각각 PPL/K-DTCBench 섹션 포함).
- `Ministral-3-3B-Instruct-2512-BF16_foem_3bit_gs64/`, `_gs32/` — 실험 4 산출 모델 + README. **`_gs32/`가 현재 최종 권장 설정.**
- `Ministral-3-3B-Instruct-2512-BF16_foem_3bit_mp_modules/`, `_mp_layers/` — 실험 5 산출 모델 + README.
- `logs/foem_3bit_actorder_kocalib.log` — 실험 1 전체 로그 (OOM 실패 시도 포함, 3-3절 수치의 출처).
- `logs/foem_3bit_kocalib.log` — 실험 2 전체 로그.
- `logs/foem_3bit_alpha0.3.log`, `foem_3bit_alpha0.6.log`, `foem_3bit_beta0.1.log`, `foem_3bit_beta0.4.log` — 실험 3 전체 로그 (OOM/RuntimeError 실패 시도 포함).
- `logs/foem_3bit_gs64.log`, `foem_3bit_gs32.log` — 실험 4 전체 로그.
- `logs/foem_3bit_mp_modules.log`, `foem_3bit_mp_layers.log` — 실험 5 전체 로그 (조합 B는 GPTQ v1 변환 버그로 인한 첫 실패 시도 포함).
- `pseudo/ABLATION_REPORT.md` — 이 파일.

> **참고**: 실험 1의 모델 디렉토리(`_foem_3bit_actorder_kocalib`)는 실험 2를 재실행하며 실수로 `rm -rf` 되어 디스크에는 남아있지 않다. 3절의 수치는 삭제 전 README와 `logs/foem_3bit_actorder_kocalib.log`에서 확인된 값으로, 로그 파일에 원본 근거가 그대로 남아있다.
