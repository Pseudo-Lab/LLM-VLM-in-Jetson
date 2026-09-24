"""모델 적재 + 추론 벤치 공용 로직 (데스크톱 evaluate.py / Jetson 스크립트 공유).

측정 항목:
  - peak_vram_gb : 추론 중 최대 GPU 메모리 (torch.cuda.max_memory_allocated 기준)
  - latency_s    : 샘플당 평균 generate 지연
  - tokens_per_s : 초당 생성 토큰 수(throughput)
  - accuracy     : (정답 필드가 있을 때) 근사 정확도 — 아래 한계 주석 참고

모델 종류(kind):
  - mllama_fp16 : 원본 Llama 3.2 Vision (fp16/bf16) — 양자화 기준선
  - gptq        : GPTQModel 로 저장한 4bit 체크포인트
  - smolvlm     : SmolVLM2 등 ImageTextToText 계열 (엣지 참고선)

주의: 실제 모델/하드웨어로 1회 돌려 검증 필요. 모델별 processor/generate 규격
차이가 있어 OOM·키 이름은 실행 시 미세 조정될 수 있음.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, asdict

import torch
from transformers import AutoProcessor


# ---------------------------------------------------------------- 모델 로딩

def load_model(kind: str, path: str, token: str | None = None):
    """kind 에 맞춰 (model, processor) 반환. 모델은 GPU 에 적재된 상태."""
    if kind == "gptq":
        import mllama_compat  # noqa: F401  gptqmodel×transformers5 보정 (선 import 필수)

        from gptqmodel import GPTQModel

        # Windows 데스크톱엔 triton/컴파일러가 없어 커널 자동선택이 실패한다.
        # torch 커널은 느리지만 어디서나 동작 (Jetson 은 Phase 2 에서 재검토).
        import sys as _sys
        backend = "torch" if _sys.platform == "win32" else "auto"
        model = GPTQModel.load(path, backend=backend)
        processor = AutoProcessor.from_pretrained(path)
        return model, processor

    if kind == "mllama_fp16":
        from transformers import MllamaForConditionalGeneration

        model = MllamaForConditionalGeneration.from_pretrained(
            path, torch_dtype=torch.bfloat16, device_map="auto", token=token
        )
        processor = AutoProcessor.from_pretrained(path, token=token)
        return model, processor

    if kind in ("nf4", "nf4_lmhead"):
        # SplitQ(2605.19929) Table 13 근거: 비전 인코더도 4bit 양자화해도 성능 유지.
        # bnb NF4 는 vision/cross-attn 포함 전체 Linear 를 4bit 로 적재 — fp16 비전경로
        # 병목(~4.5GB) 해소 실험용. GPTQ 와 달리 캘리브레이션 없는 즉석 양자화.
        # nf4_lmhead: 기본 스킵되는 lm_head(bf16 ~1GB)까지 4bit — skip 목록을 비운다.
        from transformers import BitsAndBytesConfig, MllamaForConditionalGeneration

        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            llm_int8_skip_modules=[] if kind == "nf4_lmhead" else None,
        )
        model = MllamaForConditionalGeneration.from_pretrained(
            path, quantization_config=bnb, device_map="auto", token=token
        )
        processor = AutoProcessor.from_pretrained(path, token=token)
        return model, processor

    if kind == "smolvlm":
        from transformers import AutoModelForImageTextToText

        model = AutoModelForImageTextToText.from_pretrained(
            path, torch_dtype=torch.bfloat16, device_map="auto", token=token
        )
        processor = AutoProcessor.from_pretrained(path, token=token)
        return model, processor

    raise ValueError(f"알 수 없는 kind: {kind!r} (mllama_fp16|gptq|nf4|smolvlm)")


def shrink_max_tiles(model, processor, n: int) -> None:
    """mllama 의 이미지 타일 상한을 4→n 으로 줄여 비전 활성값 메모리를 절감.

    프로세서만 바꾸면 안 된다: 타일 위치 임베딩 3개가 4타일 고정 크기라
    1타일 입력이 브로드캐스트로 4타일이 돼 reshape 이 깨진다. 임베딩 가중치를
    앞 n개 타일 분량으로 슬라이스하고 각 모듈의 max_num_tiles 를 갱신한다.
    (타일 임베딩은 양자화 대상이 아니라 bf16 그대로 — 로드 후 수술 안전)
    """
    core = getattr(model, "model", model)  # transformers5: ForConditionalGeneration.model 안에 vision_model
    vm = core.vision_model
    h = vm.config.hidden_size
    for m in (vm.pre_tile_positional_embedding, vm.post_tile_positional_embedding):
        w = m.embedding.weight.data
        m.embedding.weight.data = w.reshape(w.shape[0], m.max_num_tiles, h)[:, :n].reshape(w.shape[0], n * h)
        m.max_num_tiles = n
    g = vm.gated_positional_embedding
    w = g.tile_embedding.weight.data
    g.tile_embedding.weight.data = (
        w.reshape(w.shape[0], g.max_num_tiles, g.num_patches, h)[:, :n].reshape(w.shape[0], -1)
    )
    g.max_num_tiles = n
    vm.max_num_tiles = n
    core.max_num_tiles = n
    model.config.vision_config.max_num_tiles = n
    processor.image_processor.max_image_tiles = n


def _model_device(model) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------- 정확도(근사)

def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()


def _is_correct(generated: str, answer) -> bool:
    """근사 정확도: 생성문이 정답(또는 정답 리스트 중 하나)을 포함하면 정답 처리.

    한계: 자유 생성 + 정규화 매칭이라 표준 VQA accuracy 와 다를 수 있음.
    엄밀 비교가 필요하면 데이터셋별 공식 평가 프로토콜로 교체할 것.
    """
    g = _norm(generated)
    answers = answer if isinstance(answer, list) else [answer]
    return any(_norm(str(a)) and _norm(str(a)) in g for a in answers)


# ---------------------------------------------------------------- 벤치 본체

@dataclass
class BenchResult:
    name: str
    n: int
    peak_vram_gb: float
    latency_s: float
    tokens_per_s: float
    accuracy: float | None  # 정답 필드 없으면 None

    def as_dict(self) -> dict:
        return asdict(self)


def _build_inputs(processor, image, question: str, device):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": question},
            ],
        }
    ]
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(images=image, text=prompt, return_tensors="pt")
    return inputs.to(device)


def run_benchmark(
    model,
    processor,
    samples: list[dict],
    name: str,
    max_new_tokens: int = 64,
) -> BenchResult:
    """samples: [{"image":PIL, "question":str, "answer":str|list|None}, ...]"""
    device = _model_device(model)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    total_time, total_gen_tokens, correct, scored = 0.0, 0, 0, 0

    model.eval()
    for s in samples:
        inputs = _build_inputs(processor, s["image"], s["question"], device)
        in_len = inputs["input_ids"].shape[-1]

        t0 = time.perf_counter()
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        total_time += time.perf_counter() - t0

        gen_ids = out[0][in_len:]
        total_gen_tokens += int(gen_ids.shape[-1])

        ans = s.get("answer")
        if ans is not None:
            text = processor.decode(gen_ids, skip_special_tokens=True)
            scored += 1
            correct += int(_is_correct(text, ans))

    n = len(samples)
    peak = (torch.cuda.max_memory_allocated() / 1e9) if torch.cuda.is_available() else 0.0
    return BenchResult(
        name=name,
        n=n,
        peak_vram_gb=round(peak, 2),
        latency_s=round(total_time / max(n, 1), 3),
        tokens_per_s=round(total_gen_tokens / total_time, 1) if total_time else 0.0,
        accuracy=round(correct / scored, 4) if scored else None,
    )
