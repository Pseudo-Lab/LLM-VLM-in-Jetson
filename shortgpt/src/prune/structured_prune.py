"""구조적 width 프루닝 (수동 텐서 슬라이싱, Minitron식).

torch-pruning의 auto dependency-graph는 transformers 5.x의 Llama forward를 추적할 때
폭주(무한루프)하므로 사용하지 않는다. 대신 결합구조가 단순한 **MLP intermediate
차원**을 활성값(또는 weight magnitude) 중요도 순으로 직접 슬라이싱한다.

- 마스킹(0 채움)이 아니라 텐서를 실제로 잘라 **차원을 축소** → llama.cpp(dense)·GGUF에서
  바로 메모리 절감으로 직결된다.
- MLP 결합구조: gate/up_proj의 **출력채널**과 down_proj의 **입력채널**이 같은
  intermediate 차원 → 세 가중치를 같은 keep 인덱스로 자르면 일관성 보장.
- Llama식(분리 gate_proj/up_proj)과 Phi-3/Phi-4식(fused gate_up_proj) 둘 다 지원.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from src.common.param_stats import count_parameters


def _slice_linear(linear: nn.Linear, idx: torch.Tensor, dim: int) -> nn.Linear:
    """linear의 weight(+bias)를 dim(0=출력채널, 1=입력채널)에서 idx만 남긴 새 Linear."""
    idx = idx.to(linear.weight.device)  # 중요도가 GPU, 가중치가 CPU여도 일관되게
    w = linear.weight.data.index_select(dim, idx)
    out_f, in_f = (idx.numel(), linear.in_features) if dim == 0 else (linear.out_features, idx.numel())
    new = nn.Linear(in_f, out_f, bias=linear.bias is not None, device=w.device, dtype=w.dtype)
    new.weight.data.copy_(w)
    if linear.bias is not None:
        # 출력채널(dim 0)을 자를 때만 bias도 같이 슬라이싱
        new.bias.data.copy_(linear.bias.data.index_select(0, idx) if dim == 0 else linear.bias.data)
    return new


def _keep_indices(importance: torch.Tensor, ratio: float) -> torch.Tensor | None:
    """중요도 상위 (1-ratio) 비율의 인덱스를 정렬해 반환. 줄일 게 없으면 None."""
    total = importance.shape[0]
    keep_n = max(1, round(total * (1.0 - ratio)))
    if keep_n >= total:
        return None
    return torch.argsort(importance, descending=True)[:keep_n].sort().values


def _prune_llama_mlp(mlp: nn.Module, ratio: float, scores: dict | None) -> int:
    """분리형 gate_proj/up_proj/down_proj MLP 슬라이싱. 줄인 뉴런 수 반환."""
    inter = mlp.up_proj.out_features
    imp = None
    if scores:
        for proj in (mlp.up_proj, mlp.gate_proj):
            s = scores.get(proj)
            if s is not None and s.shape[0] == inter:
                imp = s.float()
                break
    if imp is None:  # 폴백: gate+up 가중치 행 L2 norm
        imp = mlp.gate_proj.weight.data.float().norm(dim=1) + mlp.up_proj.weight.data.float().norm(dim=1)

    keep = _keep_indices(imp, ratio)
    if keep is None:
        return 0
    mlp.gate_proj = _slice_linear(mlp.gate_proj, keep, dim=0)
    mlp.up_proj = _slice_linear(mlp.up_proj, keep, dim=0)
    mlp.down_proj = _slice_linear(mlp.down_proj, keep, dim=1)
    return inter - keep.numel()


def _prune_fused_mlp(mlp: nn.Module, ratio: float, scores: dict | None) -> int:
    """Phi-3/Phi-4식 fused gate_up_proj([2*inter, hidden]) MLP 슬라이싱.

    gate_up_proj의 앞쪽 inter개 행=gate, 뒤쪽 inter개 행=up. 같은 뉴런 집합을
    두 구간에서 동일하게 남겨야 한다.
    """
    inter = mlp.down_proj.in_features
    w = mlp.gate_up_proj.weight.data.float()
    s = scores.get(mlp.gate_up_proj) if scores else None
    if s is not None and s.shape[0] == 2 * inter:
        # gate_up_proj 출력은 [gate(inter) | up(inter)] → 같은 뉴런 i의 gate·up
        # 활성값 점수(i, i+inter)를 합산해 intermediate 뉴런 중요도로 (활성값 기반)
        s = s.float()
        imp = s[:inter] + s[inter:]
    elif s is not None and s.shape[0] == inter:
        imp = s.float()
    else:  # 폴백: gate 구간 + up 구간 weight 행 L2 norm
        imp = w[:inter].norm(dim=1) + w[inter:].norm(dim=1)

    keep = _keep_indices(imp, ratio)
    if keep is None:
        return 0
    fused_keep = torch.cat([keep, keep + inter])  # gate 행 + up 행
    mlp.gate_up_proj = _slice_linear(mlp.gate_up_proj, fused_keep, dim=0)
    mlp.down_proj = _slice_linear(mlp.down_proj, keep, dim=1)
    return inter - keep.numel()


def _prune_mlp(mlp: nn.Module, ratio: float, scores: dict | None) -> int:
    if hasattr(mlp, "gate_proj") and hasattr(mlp, "up_proj") and hasattr(mlp, "down_proj"):
        return _prune_llama_mlp(mlp, ratio, scores)
    if hasattr(mlp, "gate_up_proj") and hasattr(mlp, "down_proj"):
        return _prune_fused_mlp(mlp, ratio, scores)
    raise NotImplementedError(f"지원하지 않는 MLP 구조: {type(mlp).__name__}")


def prune_width(
    model: nn.Module,
    example_inputs: torch.Tensor | None = None,
    ratio: float = 0.3,
    importance_scores: dict[nn.Module, torch.Tensor] | None = None,
):
    """MLP intermediate를 구조적으로 슬라이싱하고 (pruned_model, info) 반환.

    Parameters
    ----------
    model: HF causal LM (model.model.layers[*].mlp 구조).
    example_inputs: 인터페이스 호환용. 수동 슬라이싱은 그래프 추적이 불필요해 사용하지 않음.
    ratio: intermediate 차원 감축 비율(0~1).
    importance_scores: {module: per-output-channel 중요도}. 없으면 weight magnitude 폴백.
    """
    params_before = count_parameters(model)

    # 슬라이싱은 GPU 연산이 불필요한 텐서 인덱싱이고, 대형 모델(14B)은 GPU 메모리에
    # 빠듯해 OOM이 난다 → CPU에서 수행(인스턴스 RAM은 넉넉). 중요도 수집은 이미 끝남.
    model.to("cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    for layer in model.model.layers:
        _prune_mlp(layer.mlp, ratio, importance_scores)

    # config의 intermediate_size를 실제 슬라이싱 결과로 갱신(저장/일관성)
    new_inter = model.model.layers[0].mlp.down_proj.in_features
    if hasattr(model.config, "intermediate_size"):
        model.config.intermediate_size = new_inter

    params_after = count_parameters(model)
    info = {
        "params_before": params_before,
        "params_after": params_after,
        "ratio_actual": 1.0 - params_after / params_before,
        "ratio_target": ratio,
    }
    return model, info
