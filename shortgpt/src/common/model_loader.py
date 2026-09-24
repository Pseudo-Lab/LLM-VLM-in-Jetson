"""HuggingFace causal LM + 토크나이저 로딩."""
from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

_DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def resolve_dtype(name: str) -> torch.dtype:
    if name not in _DTYPES:
        raise ValueError(f"지원하지 않는 dtype: {name} (가능: {list(_DTYPES)})")
    return _DTYPES[name]


def load_model_and_tokenizer(model_name: str, dtype: str = "bfloat16", device: str = "cuda"):
    """HF 모델·토크나이저를 로딩하여 (model, tokenizer) 반환.

    transformers 5.x는 `dtype=` 인자를 사용한다(과거 `torch_dtype=`의 후속).
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=resolve_dtype(dtype),
        trust_remote_code=True,
    ).to(device).eval()
    return model, tokenizer
