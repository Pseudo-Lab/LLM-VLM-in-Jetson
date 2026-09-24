"""ShortGPT식 depth(레이어) 프루닝 실행 (GPU 서버, integration).

설정(yaml)을 읽어 모델 로딩 → 한국어 보정셋 → Block Influence 측정 →
낮은 BI 레이어 제거 → 저장 → 리포트.

사용법:
    .venv/bin/python scripts/run_depth_prune.py --config configs/prune_phi4_smoke.yaml
"""
from __future__ import annotations

import argparse
import os
import sys

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.common.model_loader import load_model_and_tokenizer  # noqa: E402
from src.prune.calibration import tokenize_texts, load_korean_texts  # noqa: E402
from src.prune.depth_prune import compute_block_influence, prune_depth  # noqa: E402
from src.prune.report import format_prune_report  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default=None, help="output dir override")
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    model, tok = load_model_and_tokenizer(
        cfg["model"]["name"], dtype=cfg["model"]["dtype"], device="cuda"
    )

    texts = load_korean_texts(
        cfg["calibration"]["datasets"],
        n_samples=cfg["calibration"]["n_samples"],
        seed=cfg["calibration"]["seed"],
    )
    input_ids, attn = tokenize_texts(
        texts, tok, seq_len=cfg["calibration"]["seq_len"], return_mask=True
    )
    batch = {"input_ids": input_ids.to("cuda"), "attention_mask": attn.to("cuda")}

    bi = compute_block_influence(model, [batch])
    print("Block Influence per layer:", [round(x, 4) for x in bi.tolist()], flush=True)

    ratio = cfg["prune"].get("ratio", cfg["prune"].get("width_ratio"))  # 신키 ratio, 구키 width_ratio 폴백
    model, info = prune_depth(model, ratio=ratio, bi_scores=bi)
    print("Layers kept:", info["layers_kept"], flush=True)

    out_dir = args.out or cfg["output"]["dir"]
    os.makedirs(out_dir, exist_ok=True)
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    print(format_prune_report(info, bits=4))


if __name__ == "__main__":
    main()
