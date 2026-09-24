import torch
from src.common.param_stats import count_parameters
from src.prune.structured_prune import prune_width


def test_prune_fused_mlp_uses_activation_scores(tiny_phi3_model, example_inputs):
    """Phi-4식 fused gate_up_proj에서 활성값 중요도(2*inter)가 폴백 없이 쓰여야 한다."""
    from src.prune.importance import collect_activation_importance

    model = tiny_phi3_model
    before = count_parameters(model)
    target = "model.layers.0.mlp.gate_up_proj"
    scores_by_name = collect_activation_importance(model, [example_inputs], [target])
    # 수집된 활성값 점수는 출력채널 수(2*intermediate)
    assert scores_by_name[target].shape[0] == 2 * model.config.intermediate_size

    name_to_module = dict(model.named_modules())
    scores = {name_to_module[n]: s for n, s in scores_by_name.items()}
    pruned, info = prune_width(model, example_inputs, ratio=0.25, importance_scores=scores)

    assert count_parameters(pruned) < before
    out = pruned(example_inputs)
    assert out.logits.shape[-1] == 128


def test_prune_width_reduces_params_and_runs(tiny_model, example_inputs):
    before = count_parameters(tiny_model)
    pruned, info = prune_width(tiny_model, example_inputs, ratio=0.25)
    after = count_parameters(pruned)

    # 파라미터가 실제로 감소
    assert after < before
    assert info["params_before"] == before
    assert info["params_after"] == after
    # forward가 여전히 동작하고 vocab 차원 유지
    out = pruned(example_inputs)
    assert out.logits.shape[-1] == 128
