"""ShortGPT식 depth(레이어) 프루닝 — Block Influence 기반 (structured).

각 transformer 레이어가 잔차 스트림을 얼마나 바꾸는지를 Block Influence(BI)로 측정해,
거의 안 바꾸는(=잉여) 레이어를 통째로 제거한다. width 슬라이싱과 달리 레이어 단위로
구조를 줄이는 깊이 축 프루닝이다.

BI_i = 1 - mean_cos(레이어 i 입력 hidden, 출력 hidden)
  - 입력≈출력(코사인≈1) → 레이어가 거의 일을 안 함 → BI 낮음 → 제거 대상
  
BI = 해당 레이어가 실제로 일을 하는 정도
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.common.param_stats import count_parameters


def compute_block_influence(model: nn.Module, calib_batches: list) -> torch.Tensor:
    """레이어별 Block Influence(높을수록 중요)를 보정 배치로 측정해 반환.

    calib_batches 원소는 둘 다 허용:
      - Tensor                                  → input_ids (패딩 마스킹 없음, 하위호환)
      - dict{"input_ids", "attention_mask"}     → attention_mask로 패딩 토큰을
        코사인 집계와 forward attention에서 모두 제외 (BI 오염 방지).
    """
    layers = model.model.layers # transformer 레이어 리스트
    cos_sums = [0.0] * len(layers) # 레이어별 코사인 누적합
    counts = [0] * len(layers)
    handles = []
    cur_mask = {"m": None}  # 현재 배치의 attention_mask (hook이 참조)

    def make_hook(i: int):
        def hook(_module, inp, out):
            h_in = inp[0]
            h_out = out[0] if isinstance(out, tuple) else out
            cos = F.cosine_similarity(h_in.float(), h_out.float(), dim=-1)  # (batch, seq)
            mask = cur_mask["m"]
            if mask is not None:
                m = mask.to(cos.device).bool()  # (batch, seq) — 실제 토큰=True
                cos_sums[i] += cos[m].sum().item()
                counts[i] += int(m.sum().item())
            else:
                cos_sums[i] += cos.sum().item()
                counts[i] += cos.numel()
        return hook

    for i, layer in enumerate(layers):
        handles.append(layer.register_forward_hook(make_hook(i)))

    model.eval()
    try:
        with torch.no_grad():
            for batch in calib_batches:
                if isinstance(batch, dict):
                    input_ids = batch["input_ids"]
                    attn = batch.get("attention_mask")
                else:
                    input_ids, attn = batch, None
                cur_mask["m"] = attn
                # use_cache=False: BI 측정엔 KV 캐시 불필요(메모리 절약 → 더 큰 배치 가능)
                if attn is not None:
                    model(input_ids=input_ids, attention_mask=attn, use_cache=False)
                else:
                    model(input_ids=input_ids, use_cache=False)
    finally:
        # 캘리브 중 예외(OOM 등)가 나도 훅을 반드시 제거(다음 run 이중카운트 방지)
        for h in handles:
            h.remove()

    return torch.tensor([1.0 - cos_sums[i] / counts[i] for i in range(len(layers))]) # BI = 1 - 평균코사인


def prune_depth(model: nn.Module, ratio: float, bi_scores: torch.Tensor):
    """Block Influence가 낮은 레이어를 ratio만큼 제거하고 (model, info) 반환."""
    params_before = count_parameters(model)
    layers = model.model.layers
    n = len(layers)
    n_keep = max(1, n - round(n * ratio))

    # BI 높은(중요한) 레이어를 남기되, 원래 순서는 보존
    keep_idx = sorted(torch.argsort(bi_scores, descending=True)[:n_keep].tolist())
    new_layers = nn.ModuleList([layers[i] for i in keep_idx])

    # KV 캐시 인덱싱용 layer_idx 재부여 (중간 레이어 제거 후 0..n_keep-1)
    for new_i, layer in enumerate(new_layers):
        attn = getattr(layer, "self_attn", None)
        if attn is not None and hasattr(attn, "layer_idx"):
            attn.layer_idx = new_i

    model.model.layers = new_layers
    model.config.num_hidden_layers = n_keep

    params_after = count_parameters(model)
    return model, {
        "params_before": params_before,
        "params_after": params_after,
        "ratio_actual": 1.0 - params_after / params_before,
        "ratio_target": ratio,
        "layers_kept": keep_idx,
    }
