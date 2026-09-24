"""모델 파라미터 카운트 및 메모리 추정 유틸."""
from __future__ import annotations

import torch.nn as nn


def count_parameters(model: nn.Module) -> int:
    """학습 가능 여부와 무관하게 전체 파라미터 수를 반환."""
    return sum(p.numel() for p in model.parameters())


def estimate_memory_gb(num_params: int, bits: int) -> float:
    """주어진 비트수로 양자화했을 때 가중치 메모리(GB, 10^9 기준)."""
    return num_params * (bits / 8) / 1e9
