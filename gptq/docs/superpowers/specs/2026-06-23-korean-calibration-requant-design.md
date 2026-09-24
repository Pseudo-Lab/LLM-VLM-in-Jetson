# 한국어 혼합 캘리브레이션 재양자화 설계

작성일: 2026-06-23

## 배경 / 가설

영어 전용 캘리브(flickr30k 영어 캡션)로 양자화한 결과, 한국어 PPL 손상이 영어의
2.7배로 측정됨:

| 언어 | fp16 | GPTQ-4bit(영어캘리브) | Δ |
|------|------|------------------------|---|
| 영어 | 8.21 | 8.81 | +7.3% |
| 한국어 | 8.57 | 10.25 | **+19.6%** |

**가설**: 캘리브를 한국어 위주(70:30)로 바꾸면 한국어 PPL 손상이 줄어든다.
영어 손상은 크게 나빠지지 않는다.

## 데이터 구성

- **소스**: `piyushsinghpasi/mscoco-multilingual-30k` (ungated, 스트리밍, split=`test`)
  - 한국어 70%: `Korean` 컬럼 (COCO 캡션 한국어 번역)
  - 영어 30%: `caption` 컬럼 (같은 이미지 도메인 → 언어 외 변수 통제)
- **오염 없음**: PPL 평가셋(wikitext-en / wikipedia-ko), VLM 평가셋(textvqa)과 무중복
- **패킹**: 기존과 동일 — `pack_chars=2000`, `min_length=16`, 총 `num_samples=256`
  - 70:30 → 한국어 ~179 / 영어 ~77 패킹 샘플, seed로 셔플
- 이미지 집합이 KO/EN 간 겹쳐도 무관(텍스트 모드는 언어 문자열만 사용)

## config 스키마 변경 (`configs/gptq_config.yaml`)

`calibration`의 단일 `dataset` → 가중 멀티소스 `sources` 리스트:

```yaml
calibration:
  mode: "text"
  num_samples: 256
  min_length: 16
  pack_chars: 2000
  seed: 42
  sources:
    - dataset: "piyushsinghpasi/mscoco-multilingual-30k"
      split: "test"
      field: "Korean"
      weight: 0.7
    - dataset: "piyushsinghpasi/mscoco-multilingual-30k"
      split: "test"
      field: "caption"
      weight: 0.3
```

`sources` 미존재 시 기존 단일-dataset 동작으로 폴백(하위호환).

## 코드 변경

1. `src/calibration.py`
   - `build_text_calibration`를 가중 멀티소스 패킹으로 일반화.
   - `field`로 컬럼 직접 지정(기존 `_caption_of` 휴리스틱 대체, 폴백 유지).
   - 소스별 `weight*num_samples`개 패킹 → 합쳐서 `seed` 셔플.

2. `configs/gptq_config.yaml`
   - `calibration.sources` 추가.
   - `quantize.output_dir` → `models/Llama-3.2-11B-Vision-Instruct-gptq-4bit-kocalib`
     (기존 `...-gptq-4bit/`는 영어캘리브 기준선으로 보존 = A/B 비교).

3. `src/eval_ppl.py`
   - 하드코딩 `QUANT` 경로를 `--quant-dir` 인자로 파라미터화.
   - 새 모델로 PPL 측정 → `results/ppl_gptq_kocalib.json` 저장.

## 산출물 / 검증

- 새 모델: `models/...-gptq-4bit-kocalib/`
- 새 결과: `results/ppl_gptq_kocalib.json`
- 비교: 영어캘리브(`ppl_gptq.json`) vs 한국어캘리브(`ppl_gptq_kocalib.json`)를
  영/한 각각 나란히. fp16(`ppl_fp16.json`)은 공통 기준선.
- **성공 기준**: 한국어 Δ가 +19.6%에서 유의미하게 감소, 영어 Δ는 +7.3% 근처 유지.

## 작업 순서

1. 코드 3파일 수정
2. `python src/quantize.py` (재양자화, 수십 분~시간)
3. `python src/eval_ppl.py --model gptq --quant-dir models/...-kocalib`
4. 결과 비교 표 작성
