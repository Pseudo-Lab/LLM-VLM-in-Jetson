from src.common.param_stats import count_parameters, estimate_memory_gb


def test_tiny_model_forward(tiny_model, example_inputs):
    out = tiny_model(example_inputs)
    assert out.logits.shape == (2, 16, 128)


def test_count_parameters(tiny_model):
    n = count_parameters(tiny_model)
    assert isinstance(n, int) and n > 0


def test_estimate_memory_gb():
    # 14.7B 파라미터를 4비트로 → 약 7.35GB
    gb = estimate_memory_gb(14_700_000_000, bits=4)
    assert 7.0 < gb < 7.7
