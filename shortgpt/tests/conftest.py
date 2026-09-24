import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM


@pytest.fixture
def tiny_model():
    """Phi-4 구조(GQA, MLP gate/up/down)를 축소 재현한 결정론적 프록시 모델."""
    torch.manual_seed(0)
    config = LlamaConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,   # GQA
        max_position_embeddings=64,
    )
    model = LlamaForCausalLM(config).eval()
    return model


@pytest.fixture
def tiny_phi3_model():
    """Phi-4 실제 구조(fused gate_up_proj)를 축소 재현한 프록시 모델."""
    from transformers import Phi3Config, Phi3ForCausalLM

    torch.manual_seed(0)
    config = Phi3Config(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,   # Phi3 기본값(32000)은 tiny vocab 밖이라 재지정
    )
    return Phi3ForCausalLM(config).eval()


@pytest.fixture
def example_inputs():
    """프루닝 DependencyGraph 빌드/forward용 입력."""
    torch.manual_seed(1)
    return torch.randint(0, 128, (2, 16))
