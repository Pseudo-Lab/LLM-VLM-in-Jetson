"""GPTQ 캘리브레이션 데이터셋 준비.

핵심: GPTQModel 의 mllama(Llama 3.2 Vision) 지원은 **텍스트 레이어 한정**이다.
비전 인코더/크로스어텐션은 양자화 대상이 아니므로, 캘리브레이션은 텍스트만으로
충분하며 이것이 공식 예제(list[str])와 일치하는 안정적인 경로다.

- build_text_calibration : list[str] 반환. 기본 경로(권장).
- build_calibration      : 이미지+텍스트 processor 입력. 실험적(버전 의존).
"""
from datasets import load_dataset

from config import load_config


def _caption_of(row) -> str:
    """데이터셋 행에서 캡션 문자열 1개를 추출 (필드명/구조 차이 흡수)."""
    cap = row.get("caption") or row.get("sentence") or row.get("text")
    if isinstance(cap, list):
        cap = cap[0] if cap else ""
    return (cap or "").strip()


def build_text_calibration(num_samples: int | None = None) -> list[str]:
    """텍스트 캘리브 샘플(list[str]) 생성 — mllama 텍스트 레이어 양자화용.

    Args:
        num_samples: 샘플 수 (None 이면 config 값 사용)

    Returns:
        list[str]: GPTQModel.quantize() 에 그대로 넣을 캘리브 문장 리스트
    """
    cfg = load_config()
    ccfg = cfg["calibration"]
    n = num_samples or ccfg["num_samples"]
    min_len = ccfg.get("min_length", 0)
    # GPTQ 는 샘플당 평균 256 토큰 이상을 권장하는데 flickr30k 캡션은 ~21 토큰으로
    # 너무 짧다. 캡션 여러 개를 target_chars 까지 이어붙여 한 샘플로 패킹한다.
    target_chars = ccfg.get("pack_chars", 0)

    ds = load_dataset(ccfg["dataset"], split="test", streaming=True)

    samples: list[str] = []
    buf: list[str] = []
    for row in ds:
        if len(samples) >= n:
            break
        text = _caption_of(row)
        if len(text) < min_len:
            continue
        if target_chars <= 0:
            samples.append(text)
            continue
        buf.append(text)
        if sum(len(t) + 1 for t in buf) >= target_chars:
            samples.append(" ".join(buf))
            buf = []

    if not samples:
        raise RuntimeError(
            f"캘리브 텍스트를 한 건도 못 모았습니다 (dataset={ccfg['dataset']}). "
            "데이터셋 split/필드명을 확인하세요."
        )
    print(f"[calibration] text {len(samples)}개 준비 (dataset={ccfg['dataset']})")
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
