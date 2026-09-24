"""프루닝 before/after 리포트 포매팅."""
from __future__ import annotations

from src.common.param_stats import estimate_memory_gb


def format_prune_report(info: dict, bits: int = 4) -> str:
    """프루닝 결과 info dict를 사람이 읽는 표로 포매팅."""
    before = info["params_before"]
    after = info["params_after"]
    mem_before = estimate_memory_gb(before, bits)
    mem_after = estimate_memory_gb(after, bits)
    pct = info["ratio_actual"] * 100
    return (
        "=== Pruning Report ===\n"
        f"params: {before:,} -> {after:,} ({pct:.1f}% 감축, 목표 {info['ratio_target']*100:.0f}%)\n"
        f"메모리({bits}bit 추정): {mem_before:.2f}GB -> {mem_after:.2f}GB"
    )
