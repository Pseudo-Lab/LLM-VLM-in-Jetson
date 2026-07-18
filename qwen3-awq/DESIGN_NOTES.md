# 직접 구현 AWQ vs 공식 AutoAWQ

## 결과 (Qwen3-4B, KMMLU)

| 모델 | KMMLU acc | FP16 대비 |
|------|-----------|-----------|
| FP16 baseline | 45.81% | — |
| 공식 AutoAWQ (INT4, pileval) | 44.11% | −1.70%p |
| 직접 구현 (INT4, **kowikitext** 한국어) | **41.61%** | −4.20%p |
| 직접 구현 (INT4, pileval 영어) | 40.34% | −5.47%p |
| 직접 구현 (INT4, wikitext2 영어) | (측정 중) | — |

- INT4 로딩(40.34%)과 FP16 dequant 시뮬레이션(40.33%)이 일치 → **export 파이프라인 정상**
- 공식 대비 격차는 버그가 아니라 아래 **의도적 단순화 3가지**의 대가
- **한국어 calibration(kowikitext)이 pileval보다 +1.27%p** — 한국어 벤치마크에는
  한국어 calibration이 유리함을 확인. 특히 HUMSS(인문사회) +2.57%p로 언어 의존
  도메인에서 효과가 큼
  
## 차이점과 이유

### 1. Scale folding 없음 → 이중 양자화 (격차의 주원인)

AWQ는 `W·s`를 양자화하면 입력 쪽에 `1/s` 보정이 필요합니다.

- **공식**: `1/s`를 이전 연산(LayerNorm, 앞단 Linear)의 weight에 접음(fold) → 양자화 1회
- **우리**: `dequant(quant(W·s)) / s`를 만든 뒤 이를 **다시 INT4로 양자화** → 오차 2회 누적

**이유**: fold는 아키텍처별 레이어 연결 매핑이 필요합니다 (AutoAWQ가 모델마다 전용
클래스를 두는 이유). "아무 HF 모델에나 적용"이라는 범용성 목표를 위해 unfold를 택했습니다.

### 2. 탐색 목적함수: weight 오차 vs 출력 오차

- **공식**: calibration 입력 X를 캐시해 `‖Q(W·s)·(X/s) − W·X‖` (출력 오차) 최소화
  — activation이 큰 채널의 오차에 자동으로 가중치가 실림
- **우리**: activation 통계는 탐색 후보(`s = act_scales^α`)에만 쓰고,
  선택은 `‖dequant(quant(W·s)) − W·s‖` (weight 오차)로 함

**이유**: 출력 오차 기준은 레이어 입력 X(레이어당 수백 MB)가 필요해 공식은 블록 단위
순차 실행 인프라를 씁니다. 우리는 hook으로 채널당 통계 벡터(수 KB)만 수집하는
단순 구조를 유지했습니다 (Colab 12.67GB RAM 제약 포함).

### 3. Weight clipping 탐색 생략

공식은 그룹 max를 얼마나 잘라낼지도 grid search (outlier로 인한 해상도 낭비 방지).
핵심 알고리즘 이해에 집중하기 위해 생략 — AWQ 고유 아이디어는 아님.

## 공식과 동일한 부분

INT4 group-wise asymmetric 양자화(group 128), alpha grid search(0~1, 20 grid),
AWQ GEMM export 포맷(인터리브 패킹 `[0,2,4,6,1,3,5,7]`, gptqmodel/vLLM 호환), lm_head 제외.

## 개선 로드맵

1. **입력 서브샘플 캐시 + 출력 오차 탐색** — hook에서 레이어당 수백 행만 저장(~1GB 미만),
   범용성 유지하며 목적함수를 공식과 동일하게 (추천 다음 단계)
2. **clipping 탐색** — 범용성 유지, 구현 간단
3. **LayerNorm fold** — Llama-계열 표준 구조 한정 지원 + 미지원 모델은 폴백.
   이중 양자화 제거로 가장 큰 격차 해소, 대신 아키텍처 의존성 발생
