"""Depth pruning 실험 (ShortGPT 식 Block Influence) — self-attn 층 드롭 시 PPL 손상 측정.

mllama 는 디코더 40층 중 cross-attn 8층(3,8,13,18,23,28,33,38)이 비전 연결부라
보존하고, self-attn 32층만 드롭 후보다. 드롭은 '항등 통과(identity passthrough)'로
모사한다 — 층을 물리적으로 제거한 것과 품질이 동일하고, 인덱스/캐시 재배선 수술이
불필요하다(목적은 품질-크기 트레이드오프 측정). 실제 크기 절감은 산술 환산:
4bit 모델에서 self-attn 1층(attn+MLP) ≈ 0.11GB → K층 드롭 ≈ 11.0 - 0.11·K GB.

ShortGPT (arXiv:2403.03853): BI_i = 1 - mean_token cos(h_in, h_out).
층 입력≈출력이면 BI≈0 → '있으나 마나' → 우선 드롭 후보.

핵심 한계(정직하게): 프루닝 단독으론 4bit 모델을 크게 못 줄인다(층당 0.11GB).
8GB Jetson 적재는 [프루닝 + 잔여 fp16 전부 4bit + sub-4bit(AQLM 등)] 조합이라야.
이 실험은 그 조합의 첫 조각 — "몇 층까지 품질이 버티나"를 정량화한다.

실행: python src/prune_depth_sweep.py [--quant-dir DIR] [--drops 0 2 4 6 8] [--windows 20]
결과: results/pruning_sweep.json
"""
import argparse
import json
from pathlib import Path

import mllama_compat  # noqa: F401  gptqmodel×transformers5 보정 (선 import 필수)

import torch
from transformers import AutoTokenizer

from eval_ppl import build_token_stream, perplexity, load_model, ORIG, QUANT, SEQ_LEN

ROOT = Path(__file__).resolve().parent.parent
CROSS = {3, 8, 13, 18, 23, 28, 33, 38}   # mllama cross-attn 층 = 보존
GB_PER_LAYER_4BIT = 0.11                  # 4bit self-attn 층 1개 ≈ 0.11GB (컴포넌트 실측 기반)
BASE_GB = 11.01


def get_decoder_layers(model):
    """language_model 디코더 layers(ModuleList, len=40) 를 찾아 반환."""
    best = None
    for name, mod in model.named_modules():
        if (isinstance(mod, torch.nn.ModuleList) and len(mod) == 40
                and "language_model" in name and name.endswith("layers")):
            best = mod
    if best is None:  # 폴백: 길이 40 ModuleList 아무거나
        for name, mod in model.named_modules():
            if isinstance(mod, torch.nn.ModuleList) and len(mod) == 40:
                best = mod
    assert best is not None, "디코더 layers(len=40) 를 못 찾음"
    return best


@torch.no_grad()
def measure_bi(model, layers, stream, device, n_windows):
    """각 층의 Block Influence = mean(1 - cos(input, output)) 누적 측정."""
    sums = [0.0] * len(layers)
    cnts = [0] * len(layers)

    def mk(i):
        def hook(mod, args, kwargs, out):
            h_in = args[0] if args else kwargs.get("hidden_states")
            h_out = out[0] if isinstance(out, (tuple, list)) else out
            cos = torch.nn.functional.cosine_similarity(
                h_in.float(), h_out.float(), dim=-1)   # [B, T]
            sums[i] += (1.0 - cos).sum().item()
            cnts[i] += cos.numel()
        return hook

    handles = [l.register_forward_hook(mk(i), with_kwargs=True)
               for i, l in enumerate(layers)]
    n = min(n_windows, len(stream) // SEQ_LEN)
    for w in range(n):
        ids = stream[w * SEQ_LEN:(w + 1) * SEQ_LEN].unsqueeze(0).to(device)
        model(input_ids=ids)
        print(f"  [bi] window {w+1}/{n}", flush=True)
    for h in handles:
        h.remove()
    return [sums[i] / max(cnts[i], 1) for i in range(len(layers))]


def patch_identity(layer):
    """층 forward 를 입력 그대로 반환하도록 교체.

    mllama self-attn 디코더 층은 bare 텐서를 반환하고 루프가
    `hidden_states = decoder_layer(...)` 로 바로 받는다 → identity 도 bare 텐서 반환.
    """
    orig = layer.forward

    def fwd(hidden_states, *a, **k):
        return hidden_states

    layer.forward = fwd
    return orig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["fp16", "gptq"], default="gptq",
                    help="fp16=원본에 프루닝 먼저(양자화 전), gptq=이미 4bit인 모델에 프루닝")
    ap.add_argument("--quant-dir",
                    default=str(ROOT / "models"
                               / "Llama-3.2-11B-Vision-Instruct-gptq-4bit-kocalib"))
    ap.add_argument("--drops", type=int, nargs="+", default=[0, 2, 4, 6, 8])
    ap.add_argument("--bi-windows", type=int, default=8)
    ap.add_argument("--windows", type=int, default=20, help="sweep PPL 윈도우 수/언어")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(str(ORIG))
    streams = {lang: build_token_stream(tok, lang, args.windows * SEQ_LEN + 4096)
               for lang in ("ko", "en")}

    print(f"[load] {args.model} {args.quant_dir if args.model == 'gptq' else ORIG}", flush=True)
    model, device = load_model(args.model, Path(args.quant_dir))
    model.config.use_cache = False
    try:
        model.config.text_config.use_cache = False
    except AttributeError:
        pass

    layers = get_decoder_layers(model)
    print(f"[ok] decoder layers = {len(layers)}", flush=True)

    # 1) Block Influence 측정 (KO 텍스트)
    bi = measure_bi(model, layers, streams["ko"], device, args.bi_windows)
    cand = [(i, bi[i]) for i in range(len(layers)) if i not in CROSS]
    cand.sort(key=lambda x: x[1])               # BI 오름차순 = 드롭 우선
    drop_order = [i for i, _ in cand]
    print("[bi] self-attn 층 BI 오름차순(드롭 우선):", flush=True)
    for i, v in cand:
        print(f"    layer {i:2d}  BI={v:.4f}", flush=True)

    # 2) K층 드롭 sweep → PPL
    sweep = []
    for k in args.drops:
        to_drop = drop_order[:k]
        originals = {i: patch_identity(layers[i]) for i in to_drop}
        # ponytail: proj_size_gb 산술 환산은 4bit 전제라 fp16 스윕에선 의미 없음 → None
        proj = round(BASE_GB - GB_PER_LAYER_4BIT * k, 2) if args.model == "gptq" else None
        row = {"k": k, "dropped_layers": sorted(to_drop), "proj_size_gb": proj}
        for lang in ("ko", "en"):
            sub = streams[lang][:args.windows * SEQ_LEN]
            row[lang] = perplexity(model, sub, device)["ppl"]
        for i, orig in originals.items():       # 복원
            layers[i].forward = orig
        print(f"[sweep] k={k} drop={row['dropped_layers']} "
              f"KO_ppl={row['ko']:.3f} EN_ppl={row['en']:.3f} "
              f"~{row['proj_size_gb']}GB", flush=True)
        sweep.append(row)

    tag = "" if args.model == "gptq" else f"_{args.model}"
    out = ROOT / "results" / f"pruning_sweep{tag}.json"
    out.write_text(json.dumps({
        "model": args.model,
        "quant_dir": args.quant_dir if args.model == "gptq" else str(ORIG),
        "block_influence": {str(i): round(bi[i], 5) for i in range(len(layers))},
        "drop_order_self_attn": drop_order,
        "sweep": sweep,
        "note": "proj_size_gb = 4bit 모델 산술 환산(층당 0.11GB). "
                "프루닝 단독으론 8GB 미달; 전체4bit+sub-4bit 조합 필요.",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[done] {out}")


if __name__ == "__main__":
    main()
