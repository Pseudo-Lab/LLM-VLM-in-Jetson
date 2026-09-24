"""K-DTCBench 평가 — 한국어 Document/Table/Chart 4지선다 VQA.

NCSOFT/K-DTCBench (CC-BY-NC-4.0)
- 240문제 (document/table/chart 각 80문제)
- 한국어 네이티브 이미지, 4지선다 MCQA
- 정확매칭 채점 → LLM judge 불필요, 재현 가능

비교 목적:
  fp16 기준선 vs GPTQ-4bit (kocalib) — 양자화 손상을 한국어 VLM 태스크로 측정

실행:
  python src/eval_kdtcbench.py --model fp16
  python src/eval_kdtcbench.py --model gptq --quant-dir models/Llama-3.2-11B-Vision-Instruct-gptq-4bit-kocalib
결과: results/kdtcbench_<model>.json
"""
import argparse
import json
import re
import time
from pathlib import Path

import mllama_compat  # noqa: F401

import torch
from datasets import load_dataset

from config import load_config, hf_token, resolve
from benchmark import load_model, shrink_max_tiles
from prune_depth_sweep import get_decoder_layers, patch_identity

ROOT = Path(__file__).resolve().parent.parent
ORIG = ROOT / "models" / "Llama-3.2-11B-Vision-Instruct"
DROP_ORDER_SELF_ATTN = [
    31, 30, 34, 32, 29, 27, 35, 25, 26, 36, 24, 22, 21, 14, 20, 12,
    16, 37, 19, 15, 17, 11, 10, 9, 7, 4, 6, 2, 5, 1, 39, 0,
]


_PROMPT_TMPL = """\
다음 이미지를 보고 질문에 대한 올바른 답을 A, B, C, D 중 하나만 선택하세요.

질문: {question}

A. {a}
B. {b}
C. {c}
D. {d}

정답 (A/B/C/D 중 하나만):"""


def _build_inputs(processor, image, question: str, choices: dict, device):
    prompt = _PROMPT_TMPL.format(
        question=question,
        a=choices["a"], b=choices["b"], c=choices["c"], d=choices["d"],
    )
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(images=image, text=text, return_tensors="pt")
    return inputs.to(device)


def _extract_answer(text: str) -> str | None:
    """모델 출력에서 A/B/C/D 추출. 첫 등장 알파벳 우선."""
    text = text.strip()
    m = re.search(r"\b([A-D])\b", text.upper())
    return m.group(1) if m else None


def _model_device(model) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_drop_layers(prune_k: int) -> list[int]:
    if prune_k < 0 or prune_k > len(DROP_ORDER_SELF_ATTN):
        raise ValueError(f"prune_k must be between 0 and {len(DROP_ORDER_SELF_ATTN)}")
    return DROP_ORDER_SELF_ATTN[:prune_k]


def default_out_tag(model: str, prune_k: int) -> str:
    return f"{model}_prune_k{prune_k}" if prune_k else model


def apply_depth_pruning(model, prune_k: int) -> list[int]:
    drop_layers = resolve_drop_layers(prune_k)
    if not drop_layers:
        return []
    layers = get_decoder_layers(model)
    for idx in drop_layers:
        patch_identity(layers[idx])
    return drop_layers


def run_eval(model, processor, num_samples: int | None = None,
             per_category: int | None = None) -> dict:
    """K-DTCBench 전체 평가. 카테고리별 + 전체 정확도 반환.

    per_category: 지정 시 각 카테고리(document/table/chart)에서 N개씩 균등 추출.
                  데이터셋이 카테고리 순으로 정렬돼 있어 num_samples는 앞쪽만 집어
                  특정 카테고리에 쏠리므로, 공정한 기준선엔 per_category 사용.
    """
    ds = load_dataset("NCSOFT/K-DTCBench", split="test")
    if per_category:
        by_cat: dict[str, list[int]] = {}
        for idx, c in enumerate(ds["category"]):
            by_cat.setdefault(c, []).append(idx)
        sel = [i for c in sorted(by_cat) for i in by_cat[c][:per_category]]
        ds = ds.select(sel)
    elif num_samples:
        ds = ds.select(range(min(num_samples, len(ds))))

    device = _model_device(model)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    cats: dict[str, dict] = {}  # category → {correct, total, latency_sum}
    total_correct, total_n = 0, 0
    t_start = time.time()

    model.eval()
    for i, row in enumerate(ds):
        cat = row["category"]
        if cat not in cats:
            cats[cat] = {"correct": 0, "total": 0, "latency_sum": 0.0}

        choices = {"a": row["choice_a"], "b": row["choice_b"],
                   "c": row["choice_c"], "d": row["choice_d"]}
        inputs = _build_inputs(processor, row["image"], row["question"], choices, device)

        t0 = time.perf_counter()
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=8, do_sample=False)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        lat = time.perf_counter() - t0

        in_len = inputs["input_ids"].shape[-1]
        gen_text = processor.decode(out[0][in_len:], skip_special_tokens=True)
        pred = _extract_answer(gen_text)
        correct = int(pred == row["answer"]) if pred else 0

        cats[cat]["correct"] += correct
        cats[cat]["total"] += 1
        cats[cat]["latency_sum"] += lat
        total_correct += correct
        total_n += 1

        if (i + 1) % 20 == 0 or (i + 1) == len(ds):
            elapsed = time.time() - t_start
            acc = total_correct / total_n
            print(f"  [{i+1}/{len(ds)}] acc={acc:.3f}  ({elapsed:.0f}s)", flush=True)

    peak_vram = (torch.cuda.max_memory_allocated() / 1e9) if torch.cuda.is_available() else 0.0

    by_cat = {
        c: {
            "accuracy": round(v["correct"] / v["total"], 4),
            "correct": v["correct"],
            "total": v["total"],
            "avg_latency_s": round(v["latency_sum"] / v["total"], 3),
        }
        for c, v in cats.items()
    }
    return {
        "total_accuracy": round(total_correct / total_n, 4),
        "total_correct": total_correct,
        "total_n": total_n,
        "by_category": by_cat,
        "peak_vram_gb": round(peak_vram, 2),
        "elapsed_s": round(time.time() - t_start, 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["fp16", "gptq", "nf4", "nf4_lmhead"], required=True)
    ap.add_argument("--quant-dir",
                    default=str(ROOT / "models" / "Llama-3.2-11B-Vision-Instruct-gptq-4bit-kocalib"),
                    help="gptq 모델 경로")
    ap.add_argument("--num-samples", type=int, default=None,
                    help="평가 샘플 수 제한 (앞에서부터, None=전체 240)")
    ap.add_argument("--per-category", type=int, default=None,
                    help="카테고리별 N개씩 균등 추출 (공정한 기준선용)")
    ap.add_argument("--prune-k", type=int, default=0,
                    help="Block Influence 순서대로 self-attn K층을 identity passthrough 처리")
    ap.add_argument("--max-tiles", type=int, default=0,
                    help="이미지 타일 수 상한 (기본 0=모델 기본값 4). 비전 활성값 메모리 절감 실험용")
    ap.add_argument("--out-tag", default="", help="결과 파일명 접미사")
    args = ap.parse_args()

    token = hf_token()
    path = args.quant_dir if args.model == "gptq" else str(ORIG)

    print(f"[load] {args.model} 모델 로드 중… ({path})", flush=True)
    kind = {"fp16": "mllama_fp16", "gptq": "gptq", "nf4": "nf4", "nf4_lmhead": "nf4_lmhead"}[args.model]
    model, processor = load_model(kind, path, token=token)
    if hasattr(model, "get_memory_footprint"):
        print(f"[load] weights footprint: {model.get_memory_footprint() / 1e9:.2f}GB", flush=True)
    if args.max_tiles:
        shrink_max_tiles(model, processor, args.max_tiles)
        print(f"[load] max_image_tiles={args.max_tiles}", flush=True)
    print("[load] OK", flush=True)

    # ponytail: patch_identity 는 KV캐시 갱신을 안 해서 generate()의 use_cache=True 와
    # 충돌한다(prune_depth_sweep.py 가 이미 겪고 끈 문제와 동일) — 프루닝 유무와 무관하게 끈다.
    model.config.use_cache = False
    try:
        model.config.text_config.use_cache = False
    except AttributeError:
        pass

    dropped_layers = apply_depth_pruning(model, args.prune_k)
    if dropped_layers:
        print(f"[prune] k={args.prune_k} dropped_layers={dropped_layers}", flush=True)

    print("[eval] K-DTCBench 평가 시작…", flush=True)
    results = run_eval(model, processor, args.num_samples, args.per_category)

    tag = args.out_tag if args.out_tag else default_out_tag(args.model, args.prune_k)
    suffix = f"_{tag}"
    out = ROOT / "results" / f"kdtcbench{suffix}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(
        {
            "model": args.model,
            "quant_dir": args.quant_dir,
            "prune_k": args.prune_k,
            "dropped_layers": dropped_layers,
            "max_tiles": args.max_tiles or 4,
            **results,
        },
        ensure_ascii=False, indent=2
    ), encoding="utf-8")

    print(f"\n[done] {out}")
    print(f"  전체 정확도: {results['total_accuracy']:.4f} ({results['total_correct']}/{results['total_n']})")
    for cat, r in results["by_category"].items():
        print(f"  {cat}: {r['accuracy']:.4f} ({r['correct']}/{r['total']})")
    print(f"  peak VRAM: {results['peak_vram_gb']}GB  elapsed: {results['elapsed_s']}s")


if __name__ == "__main__":
    main()
