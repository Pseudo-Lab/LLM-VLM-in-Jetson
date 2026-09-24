"""활성값 기반 구조적 프루닝 중요도(Minitron 방식)."""
from __future__ import annotations

import torch
import torch.nn as nn


def collect_activation_importance(
    model: nn.Module,
    calib_batches: list[torch.Tensor],
    target_module_names: list[str],
) -> dict[str, torch.Tensor]:
    """대상 모듈 출력 채널별 활성값 L2 중요도를 보정 배치로 누적해 반환.

    importance[ch] = sqrt(mean_over_tokens(activation[..., ch] ** 2))
    """
    name_to_module = dict(model.named_modules())
    sums: dict[str, torch.Tensor] = {}
    counts: dict[str, int] = {}
    handles = []

    def make_hook(name: str):
        def hook(_module, _inp, out):
            act = out.detach().float()
            flat = act.reshape(-1, act.shape[-1])  # (tokens, channels)
            sq = (flat ** 2).sum(dim=0)
            sums[name] = sums.get(name, torch.zeros_like(sq)) + sq
            counts[name] = counts.get(name, 0) + flat.shape[0]
        return hook

    for name in target_module_names:
        handles.append(name_to_module[name].register_forward_hook(make_hook(name)))

    model.eval()
    with torch.no_grad():
        for batch in calib_batches:
            model(batch)

    for h in handles:
        h.remove()

    return {name: torch.sqrt(sums[name] / counts[name]) for name in target_module_names}
