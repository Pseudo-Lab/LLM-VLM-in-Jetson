"""GPTQ 캘리브레이션 데이터셋 준비.

핵심: GPTQModel 의 mllama(Llama 3.2 Vision) 지원은 **텍스트 레이어 한정**이다.
비전 인코더/크로스어텐션은 양자화 대상이 아니므로, 캘리브레이션은 텍스트만으로
충분하며 이것이 공식 예제(list[str])와 일치하는 안정적인 경로다.

- build_text_calibration : list[str] 반환. 기본 경로(권장).
- build_calibration      : 이미지+텍스트 processor 입력. 실험적(버전 의존).

캘리브 데이터는 config 의 calibration.sources (가중 멀티소스) 로 구성한다. 예:
한국어 캡션 70% + 영어 캡션 30% 를 섞으면, 영어 전용 캘리브 대비 한국어 양자화
손상을 줄일 수 있다(언어별 PPL Δ 로 검증). sources 가 없으면 단일 dataset 폴백.
"""
import random

from datasets import load_dataset

from config import load_config


def _caption_of(row) -> str:
    """데이터셋 행에서 캡션 문자열 1개를 추출 (필드명/구조 차이 흡수)."""
    cap = row.get("caption") or row.get("sentence") or row.get("text")
    if isinstance(cap, list):
        cap = cap[0] if cap else ""
    return (cap or "").strip()


def _text_of(row, field: str | None) -> str:
    """지정 컬럼(field)에서 텍스트 1개 추출. field 가 없으면 _caption_of 휴리스틱."""
    if not field:
        return _caption_of(row)
    val = row.get(field)
    if isinstance(val, list):
        val = val[0] if val else ""
    return (val or "").strip()


def _pack_source(src: dict, n: int, min_len: int, target_chars: int) -> list[str]:
    """한 소스에서 n 개의 (패킹된) 텍스트 샘플 수집.

    GPTQ 는 샘플당 평균 256 토큰 이상을 권장하는데 캡션은 ~20 토큰으로 너무 짧다.
    target_chars 까지 캡션 여러 개를 이어붙여 한 샘플로 패킹한다.
    config 키가 있으면 load_dataset(name, config, split=...) 형태로 호출한다
    (예: wikimedia/wikipedia 의 언어 서브셋 지정).
    """
    config = src.get("config")
    split = src.get("split", "test")
    if config:
        ds = load_dataset(src["dataset"], config, split=split, streaming=True)
    else:
        ds = load_dataset(src["dataset"], split=split, streaming=True)
    field = src.get("field")

    samples: list[str] = []
    buf: list[str] = []
    for row in ds:
        if len(samples) >= n:
            break
        text = _text_of(row, field)
        if len(text) < min_len:
            continue
        if target_chars <= 0:
            samples.append(text)
            continue
        # 긴 텍스트(Wikipedia 아티클 등)를 target_chars로 절단해 OOM 방지.
        # 절단 전 전체를 버퍼에 넣으면 GPTQ Hessian 계산이 수백 GB를 요구한다.
        if len(text) > target_chars:
            text = text[:target_chars]
        buf.append(text)
        if sum(len(t) + 1 for t in buf) >= target_chars:
            samples.append(" ".join(buf))
            buf = []
    return samples


def build_text_calibration(num_samples: int | None = None) -> list[str]:
    """텍스트 캘리브 샘플(list[str]) 생성 — mllama 텍스트 레이어 양자화용.

    config.calibration.sources (가중 멀티소스) 를 읽어 소스별로 weight*num_samples
    개를 패킹한 뒤 합쳐서 seed 로 셔플한다. sources 가 없으면 단일 dataset 폴백.

    Args:
        num_samples: 총 샘플 수 (None 이면 config 값 사용)

    Returns:
        list[str]: GPTQModel.quantize() 에 그대로 넣을 캘리브 문장 리스트
    """
    cfg = load_config()
    ccfg = cfg["calibration"]
    n = num_samples or ccfg["num_samples"]
    min_len = ccfg.get("min_length", 0)
    target_chars = ccfg.get("pack_chars", 0)
    seed = ccfg.get("seed", 42)

    sources = ccfg.get("sources")
    if not sources:
        # 하위호환: 단일 dataset (영어 캡션 전용)
        sources = [{"dataset": ccfg["dataset"], "split": "test",
                    "field": None, "weight": 1.0}]

    samples: list[str] = []
    for src in sources:
        want = max(1, round(n * src.get("weight", 1.0 / len(sources))))
        got = _pack_source(src, want, min_len, target_chars)
        label = f"{src['dataset']}:{src.get('field') or 'auto'}"
        print(f"[calibration] {label} {len(got)}개 (목표 {want})")
        samples.extend(got)

    if not samples:
        raise RuntimeError(
            "캘리브 텍스트를 한 건도 못 모았습니다. "
            "calibration.sources 의 dataset/split/field 를 확인하세요."
        )
    random.Random(seed).shuffle(samples)
    print(f"[calibration] text 총 {len(samples)}개 준비 (sources={len(sources)}, seed={seed})")
    return samples


def build_calibration(processor, num_samples: int | None = None):
    """[실험적] 이미지+텍스트 멀티모달 캘리브 샘플 리스트 생성.

    주의: GPTQModel 은 mllama 의 텍스트 레이어만 양자화하므로 보통은
    build_text_calibration 으로 충분하다. 이 경로는 설치된 gptqmodel 버전의
    multimodal 입력 규격과 대조해 검증한 뒤에만 사용할 것.

    Args:
        processor: AutoProcessor (mllama). quantize.py 에서 로드해 전달.
        num_samples: 샘플 수 (None 이면 config 값 사용)

    Returns:
        list[dict]: GPTQModel.quantize() 에 넣을 캘리브 데이터
    """
    cfg = load_config()
    ccfg = cfg["calibration"]
    n = num_samples or ccfg["num_samples"]

    ds = load_dataset(ccfg["dataset"], split="test", streaming=True)

    samples = []
    for i, row in enumerate(ds):
        if len(samples) >= n:
            break
        image = row.get("image")
        # 데이터셋에 따라 caption 필드명이 다름 (flickr30k: "caption" 리스트)
        caption = row.get("caption")
        if isinstance(caption, list):
            caption = caption[0] if caption else ""
        if image is None:
            continue

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "Describe this image."},
                ],
            }
        ]
        prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(
            images=image,
            text=prompt,
            return_tensors="pt",
        )
        samples.append(inputs)

    print(f"[calibration] {len(samples)} 샘플 준비 (dataset={ccfg['dataset']})")
    return samples
