"""run_prune CLI용 헬퍼 (단위 테스트 가능한 순수 로직)."""
from __future__ import annotations

import torch.nn as nn


def select_target_modules(model: nn.Module) -> list[str]:
    """활성값 중요도를 수집할 대상 Linear(주로 MLP) 이름 목록.

    embedding·lm_head는 제외하고, MLP 경로의 Linear만 고른다.
    """
    targets = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and ".mlp." in name:
            targets.append(name)
    return targets
