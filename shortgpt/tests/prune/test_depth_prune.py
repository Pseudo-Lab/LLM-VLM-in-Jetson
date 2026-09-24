import torch

from src.common.param_stats import count_parameters
from src.prune.depth_prune import compute_block_influence, prune_depth


def test_block_influence_shape(tiny_model, example_inputs):
    bi = compute_block_influence(tiny_model, [example_inputs])
    assert bi.shape[0] == tiny_model.config.num_hidden_layers  # 2


def test_block_influence_respects_attention_mask(tiny_model, example_inputs):
    # 시퀀스 뒷부분을 패딩으로 표시한 마스크
    mask = torch.ones_like(example_inputs)
    mask[:, example_inputs.shape[1] // 2:] = 0
    bi_masked = compute_block_influence(
        tiny_model, [{"input_ids": example_inputs, "attention_mask": mask}]
    )
    bi_unmasked = compute_block_influence(tiny_model, [example_inputs])

    assert bi_masked.shape[0] == tiny_model.config.num_hidden_layers
    assert torch.isfinite(bi_masked).all()
    # 패딩을 집계/attention에서 제외하면 BI 값이 달라진다(= 마스크가 실제 반영됨)
    assert not torch.allclose(bi_masked, bi_unmasked)


def test_prune_depth_removes_layers_and_runs(tiny_model, example_inputs):
    bi = compute_block_influence(tiny_model, [example_inputs])
    before = count_parameters(tiny_model)
    pruned, info = prune_depth(tiny_model, ratio=0.5, bi_scores=bi)

    # 레이어가 실제로 줄고(2 -> 1) 파라미터 감소
    assert pruned.config.num_hidden_layers == 1
    assert len(pruned.model.layers) == 1
    assert count_parameters(pruned) < before
    # forward 정상 + vocab 차원 유지
    out = pruned(example_inputs)
    assert out.logits.shape[-1] == 128
