# 직접 구현 AWQ vs 공식 AutoAWQ

## 결과 (Qwen3-4B, KMMLU)

| 모델 | 탐색 목적함수 | clip | KMMLU acc | FP16 대비 |
|------|------|------|-----------|-----------|
| FP16 baseline | — | — | 45.81% | — |
| 공식 AutoAWQ (INT4, pileval) | 출력 오차 | ✓ | 44.11% | −1.70%p |
| **직접 구현 (kowikitext, 출력 오차)** | 출력 오차 | ✗ | **42.98%** | **−2.83%p** |
| 직접 구현 (kowikitext, 출력 오차 + clip) | 출력 오차 | ✓ | 41.53% | −4.28%p |
| 직접 구현 (kowikitext, weight 오차) | weight 오차 | ✗ | 41.73% | −4.08%p |
| 직접 구현 (kowikitext, weight 오차 + clip) | weight 오차 | ✓ | 38.51% | −7.30%p |
| 직접 구현 (pileval 영어, weight 오차) | weight 오차 | ✗ | 40.34% | −5.47%p |
| 직접 구현 (wikitext2 영어, weight 오차) | weight 오차 | ✗ | 39.14% | −6.67%p |

*(RunPod 환경 측정. kowikitext weight-오차 결과는 Colab에서 41.61%로 재현성 확인됨)*

- **로드맵 #1(출력-오차 탐색) 적용으로 공식과의 격차가 −4.08%p → −1.13%p로 좁혀짐.**
  남은 격차는 대부분 아래 차이점 #1(이중 양자화)에서 기인
- INT4 로딩(40.34%)과 FP16 dequant 시뮬레이션(40.33%)이 일치 → **export 파이프라인 정상**
- **한국어 calibration(kowikitext)이 pileval보다 +1.39%p, wikitext2보다 +2.59%p** —
  한국어 벤치마크에는 한국어 calibration이 유리함을 확인. 특히 HUMSS(인문사회)에서 효과가 큼

### 압축 효과 (VRAM, 추론 로드 기준)

| | 경량화 전 (FP16) | 경량화 후 (INT4) | 압축률 |
|---|---|---|---|
| 모델 크기 | 8.04 GB | 2.67 GB | 3.0× |
| 파라미터 수 | 4.02B | 4.02B (비트수만 16→4) | — |
  
## 차이점과 이유

### 1. Scale folding 없음 → 이중 양자화 (격차의 주원인)

AWQ는 `W·s`를 양자화하면 입력 쪽에 `1/s` 보정이 필요합니다.

- **공식**: `1/s`를 이전 연산(LayerNorm, 앞단 Linear)의 weight에 접음(fold) → 양자화 1회
- **우리**: `dequant(quant(W·s)) / s`를 만든 뒤 이를 **다시 INT4로 양자화** → 오차 2회 누적

**이유**: fold는 아키텍처별 레이어 연결 매핑이 필요합니다 (AutoAWQ가 모델마다 전용
클래스를 두는 이유). "아무 HF 모델에나 적용"이라는 범용성 목표를 위해 unfold를 택했습니다.

### 2. 탐색 목적함수: weight 오차 vs 출력 오차 (로드맵 #1로 해소됨 ✓)

- **공식**: calibration 입력 X를 캐시해 `‖Q(W·s)·(X/s) − W·X‖` (출력 오차) 최소화
  — activation이 큰 채널의 오차에 자동으로 가중치가 실림
- **초기 우리**: activation 통계는 탐색 후보(`s = act_scales^α`)에만 쓰고,
  선택은 `‖dequant(quant(W·s)) − W·s‖` (weight 오차)로 함

**해소**: 로드맵 #1에서 hook에 레이어별 입력 서브샘플(512행, fp16/CPU, 전체 ~1GB)을
캐시하는 경로를 추가하고, `--search-mode output`으로 목적함수를 공식과 동일한 출력 오차로
교체함. 블록 단위 순차 실행 인프라 없이도 서브샘플만으로 근사가 가능했고,
결과적으로 **41.73% → 42.98% (+1.25%p)** 상승.

### 3. Weight clipping 탐색 (구현 완료, 실험으로 목적함수 종속성 규명 ✓)

공식은 그룹 max를 얼마나 잘라낼지도 grid search (outlier로 인한 해상도 낭비 방지).
구현 후 실험한 결과, **clip은 어떤 목적함수 위에서 도느냐에 성패가 갈림**:

| clip 탐색 기준 | KMMLU | baseline 대비 |
|---|---|---|
| weight 오차 위에 clip | 38.51% | **−3.22%p (역효과)** |
| 출력 오차 위에 clip | 41.53% | −1.45%p |

- **weight-오차 기준 clip이 역효과인 이유**: weight 재구성 오차만 보면 그룹 내 소수의
  큰 weight(outlier)를 잘라 나머지 해상도를 높이는 게 항상 이득처럼 보인다. 그런데
  그 outlier가 바로 AWQ가 보호하려는 salient weight(큰 activation과 곱해지는 채널)라서,
  잘라내면 출력이 망가진다. 공식이 clip도 **출력 오차** 기준으로 탐색하는 이유가 이것.
- **단, 출력-오차 탐색이 이미 salient weight를 잘 보호하므로**, 그 위에 clip을 더해도
  이 케이스에선 소폭 손해(42.98% → 41.53%). **최선 조합은 "출력-오차 탐색 + clip 없음".**



## 공식과 동일한 부분

INT4 group-wise asymmetric 양자화(group 128), alpha grid search(0~1, 20 grid),
AWQ GEMM export 포맷(인터리브 패킹 `[0,2,4,6,1,3,5,7]`, gptqmodel/vLLM 호환), lm_head 제외.

## 개선 로드맵

1. ✅ **입력 서브샘플 캐시 + 출력 오차 탐색** (완료) — hook에서 레이어당 512행 저장(~1GB),
   범용성 유지하며 목적함수를 공식과 동일하게. `--search-mode output`. **+1.25%p**
2. ✅ **clipping 탐색** (완료) — 구현 후 실험으로 "목적함수 종속" 규명 (위 차이점 #3 참조).
   출력-오차 탐색이 선행돼야 의미가 있으며, 이 케이스에선 clip 없는 쪽이 최선.
3. ⬜ **LayerNorm fold** — Llama-계열 표준 구조 한정 지원 + 미지원 모델은 폴백.
   이중 양자화 제거로 남은 격차(−1.13%p)의 대부분 해소 예상, 대신 아키텍처 의존성 발생.
   **현재 남은 유일한 주요 개선 항목.**

## 실험 로그 요약

- Colab → RunPod로 실행 환경 이전 (HF 다운로드 스톨/환경 리셋 대응). 코드는 GitHub로 동기화.
- fp16 overflow 버그 수정(`ec22cef`): wikitext2 calibration의 activation outlier가
  `w·act_scales^α`를 fp16에서 inf로 만들어 scales에 NaN 전파 → KMMLU 9.87%(랜덤 이하)로 붕괴.
  양자화 계산을 fp32로, 저장만 fp16으로 바꿔 해소 (wikitext2 39.14%로 정상화).
- 로드맵 #1(출력-오차 탐색, `5703ff8`) 적용: kowikitext 42.98%로 최고 성능 달성,
  공식 AutoAWQ(44.11%)와 −1.13%p까지 근접.

