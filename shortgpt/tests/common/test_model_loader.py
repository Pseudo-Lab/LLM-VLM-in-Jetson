import pytest
from src.common.model_loader import resolve_dtype


def test_resolve_dtype():
    import torch
    assert resolve_dtype("bfloat16") == torch.bfloat16
    assert resolve_dtype("float16") == torch.float16
    assert resolve_dtype("float32") == torch.float32


def test_resolve_dtype_invalid():
    with pytest.raises(ValueError):
        resolve_dtype("int4")


@pytest.mark.integration
def test_load_real_model():
    # GPU 서버에서만: 실제 Phi-4 로딩 확인
    from src.common.model_loader import load_model_and_tokenizer
    model, tok = load_model_and_tokenizer("microsoft/phi-4", dtype="bfloat16", device="cuda")
    assert model is not None and tok is not None
