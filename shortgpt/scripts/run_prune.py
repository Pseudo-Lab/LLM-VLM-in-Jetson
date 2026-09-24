"""Phi-4 구조적 width 프루닝 실행 (GPU 서버).

설정(yaml)을 읽어 모델 로딩 → 한국어 보정셋 → 활성값 중요도 →
MLP intermediate 슬라이싱 → 저장 → 리포트까지 엮는다.

실제 대형 모델/데이터가 필요하므로 GPU 서버에서 실행한다(로컬 단위테스트 대상 아님).
사용법:
    .venv/bin/python scripts/run_prune.py --config configs/prune_phi4.yaml
"""
from __future__ import annotations

import argparse
import os
import sys

import yaml

# 'src' 패키지를 import 할 수 있도록 compression/ 루트를 경로에 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.common.model_loader import load_model_and_tokenizer  # noqa: E402
from src.prune.calibration import tokenize_texts, load_korean_texts  # noqa: E402
from src.prune.importance import collect_activation_importance  # noqa: E402
from src.prune.run_helpers import select_target_modules  # noqa: E402
from src.prune.structured_prune import prune_width  # noqa: E402
from src.prune.report import format_prune_report  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    model, tok = load_model_and_tokenizer(
        cfg["model"]["name"], dtype=cfg["model"]["dtype"], device="cuda"
    )

    # 한국어 보정셋 → 토크나이즈 (활성값 중요도 수집용)
    texts = load_korean_texts(
        cfg["calibration"]["datasets"],
        n_samples=cfg["calibration"]["n_samples"],
        seed=cfg["calibration"]["seed"],
    )
    batch = tokenize_texts(texts, tok, seq_len=cfg["calibration"]["seq_len"]).to("cuda")

    # MLP Linear 대상 → 활성값 기반 중요도 수집 → 모듈 키 dict로 변환
    target_names = select_target_modules(model)
    scores_by_name = collect_activation_importance(model, [batch], target_names)
    name_to_module = dict(model.named_modules())
    scores_by_module = {name_to_module[n]: s for n, s in scores_by_name.items()}

    model, info = prune_width(
        model,
        example_inputs=batch[:1],
        ratio=cfg["prune"].get("ratio", cfg["prune"].get("width_ratio")),
        importance_scores=scores_by_module,
    )

    out_dir = cfg["output"]["dir"]
    os.makedirs(out_dir, exist_ok=True)
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    print(format_prune_report(info, bits=4))


if __name__ == "__main__":
    main()
