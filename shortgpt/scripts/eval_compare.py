"""여러 모델의 KMMLU 정확도를 비교 측정 (GPU 서버, integration).

사용법:
    .venv/bin/python scripts/eval_compare.py --limit 20 \
        --models original=microsoft/phi-4 \
                 pruned_act=artifacts/phi4-pruned-act \
                 pruned_mag=artifacts/phi4-pruned-smoke
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.common.eval_kmmlu import run_kmmlu  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True, help="name=path 형식 (공백 구분)")
    ap.add_argument("--limit", type=int, default=None, help="태스크당 샘플 수 제한")
    args = ap.parse_args()

    rows = []
    for spec in args.models:
        name, path = spec.split("=", 1)
        print(f"[eval] {name} ({path}) ...", flush=True)
        acc = run_kmmlu(path, limit=args.limit, device="cuda")
        rows.append((name, acc))
        print(f"  -> {name}: KMMLU acc = {acc:.4f}", flush=True)

    print("\n=== KMMLU 비교 결과 ===")
    for name, acc in rows:
        print(f"{name:24s} {acc * 100:.2f}%")


if __name__ == "__main__":
    main()
