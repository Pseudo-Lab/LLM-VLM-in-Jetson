import torch
from src.prune.importance import collect_activation_importance


def test_collect_activation_importance_per_channel(tiny_model, example_inputs):
    # 각 MLP up_proj 출력 채널(intermediate_size=128)에 대한 중요도 점수
    target = "model.layers.0.mlp.up_proj"
    scores = collect_activation_importance(tiny_model, [example_inputs], [target])
    assert target in scores
    vec = scores[target]
    assert vec.shape == (128,)
    assert torch.all(vec >= 0)
