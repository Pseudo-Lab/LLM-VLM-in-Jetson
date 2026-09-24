from src.prune.run_helpers import select_target_modules


def test_select_target_modules_picks_mlp_linears(tiny_model):
    names = select_target_modules(tiny_model)
    assert "model.layers.0.mlp.up_proj" in names
    assert "model.layers.0.mlp.gate_proj" in names
    # embedding/lm_head은 제외
    assert all("embed_tokens" not in n and "lm_head" not in n for n in names)
