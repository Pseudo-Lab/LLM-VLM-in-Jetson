"""Perplexity 비교 — fp16 원본 vs GPTQ 4bit (양자화 손상도 1차 정량 지표).

양자화 논문 표준 프로토콜의 축소판: 고정 텍스트를 비겹침 윈도우로 잘라
다음-토큰 예측 NLL 을 누적, PPL = exp(총NLL/총토큰).
두 모델에 '완전히 동일한' 토큰 스트림·윈도우를 쓰므로 Δ가 곧 양자화 효과다.

주의: 텍스트 전용 입력이라 mllama 의 cross-attention 층은 skip 된다
(이미지 없으면 해당 층 continue — modeling_mllama 참조). 즉 이 PPL 은
"4bit 로 깎인 self-attn 32층 + MLP" 경로의 손상을 측정한다. 비전 경로
품질은 추후 한국어 VLM 벤치마크에서 별도 측정.

실행:
  python src/eval_ppl.py --model gptq   # 4bit (GPU)
  python src/eval_ppl.py --model fp16   # 원본 (GPU+CPU 오프로드, 느림)
결과: results/ppl_<model>.json
"""
import argparse
import json
import time
from pathlib import Path

import mllama_compat  # noqa: F401  gptqmodel×transformers5 보정 (선 import 필수)

import torch
from datasets import load_dataset
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
ORIG = ROOT / "models" / "Llama-3.2-11B-Vision-Instruct"
QUANT = ROOT / "models" / "Llama-3.2-11B-Vision-Instruct-gptq-4bit"

SEQ_LEN = 1024          # 윈도우 길이 (두 모델 동일해야 비교 성립)
TOKENS_PER_LANG = 40_000  # 언어당 평가 토큰 수 (~39 윈도우)


def build_token_stream(tokenizer, lang: str, budget: int) -> torch.Tensor:
    """언어별 고정 텍스트를 budget 토큰까지 이어붙여 1-D 텐서로."""
    if lang == "en":
        ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
        text = "\n\n".join(t for t in ds["text"] if t.strip())
        ids = tokenizer(text, return_tensors="pt").input_ids[0][:budget]
    else:  # ko
        ds = load_dataset("wikimedia/wikipedia", "20231101.ko", split="train", streaming=True)
        chunks, n = [], 0
        for row in ds:
            t = (row.get("text") or "").strip()
            if len(t) < 200:
                continue
            ids = tokenizer(t, return_tensors="pt").input_ids[0]
            chunks.append(ids)
            n += len(ids)
            if n >= budget:
                break
        ids = torch.cat(chunks)[:budget]
    print(f"[data] {lang}: {len(ids)} tokens")
    return ids


@torch.no_grad()
def perplexity(model, stream: torch.Tensor, device: str) -> dict:
    """비겹침 윈도우 NLL 누적 → PPL."""
    total_nll, total_tok = 0.0, 0
    n_windows = len(stream) // SEQ_LEN
    t0 = time.time()
    for i in range(n_windows):
        ids = stream[i * SEQ_LEN:(i + 1) * SEQ_LEN].unsqueeze(0).to(device)
        logits = model(input_ids=ids).logits.float()
        # shift: 위치 t 의 logits 로 t+1 토큰 예측
        nll = torch.nn.functional.cross_entropy(
            logits[0, :-1], ids[0, 1:], reduction="sum"
        )
        total_nll += nll.item()
        total_tok += SEQ_LEN - 1
        if (i + 1) % 5 == 0 or i == n_windows - 1:
            el = time.time() - t0
            print(f"  [ppl] window {i+1}/{n_windows}  "
                  f"running_ppl={torch.exp(torch.tensor(total_nll/total_tok)):.3f}  "
                  f"({el:.0f}s)", flush=True)
    return {"ppl": float(torch.exp(torch.tensor(total_nll / total_tok))),
            "tokens": total_tok, "windows": n_windows, "seq_len": SEQ_LEN}


def load_model(kind: str, quant_dir: Path = QUANT):
    if kind == "gptq":
        from gptqmodel import GPTQModel
        m = GPTQModel.load(str(quant_dir), backend="torch")
        return m.model, "cuda"
    if kind in ("nf4", "nf4_lmhead"):
        # bnb NF4 — vision/cross-attn 포함 전체 4bit (benchmark.py 의 nf4 와 동일 설정)
        from transformers import BitsAndBytesConfig, MllamaForConditionalGeneration
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            llm_int8_skip_modules=[] if kind == "nf4_lmhead" else None,
        )
        m = MllamaForConditionalGeneration.from_pretrained(
            str(ORIG), quantization_config=bnb, device_map="auto"
        )
        m.eval()
        return m, "cuda"
    # fp16(bf16) 원본 — 12GB VRAM 에 안 들어가므로 accelerate 가 CPU 로 오프로드
    from transformers import MllamaForConditionalGeneration
    m = MllamaForConditionalGeneration.from_pretrained(
        str(ORIG), dtype=torch.bfloat16, device_map="auto"
    )
    m.eval()
    return m, "cuda"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["fp16", "gptq", "nf4", "nf4_lmhead"], required=True)
    ap.add_argument("--quant-dir", default=str(QUANT),
                    help="gptq 모델 경로 (기본: 영어캘리브 ...-gptq-4bit)")
    ap.add_argument("--out-tag", default="",
                    help="결과 파일명 접미사. 예: kocalib → ppl_gptq_kocalib.json")
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(str(ORIG))
    streams = {lang: build_token_stream(tokenizer, lang, TOKENS_PER_LANG)
               for lang in ("en", "ko")}

    print(f"[load] {args.model} 모델 로드 중…", flush=True)
    if args.model == "gptq":
        print(f"[load] quant_dir={args.quant_dir}", flush=True)
    model, device = load_model(args.model, Path(args.quant_dir))
    print("[load] OK", flush=True)

    results = {}
    for lang, stream in streams.items():
        print(f"[eval] {args.model} / {lang}", flush=True)
        results[lang] = perplexity(model, stream, device)

    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    suffix = f"_{args.out_tag}" if args.out_tag else ""
    path = out / f"ppl_{args.model}{suffix}.json"
    json.dump({"model": args.model, "quant_dir": args.quant_dir, **results},
              open(path, "w"), indent=2)
    print(f"[done] {path}")
    for lang, r in results.items():
        print(f"  {lang}: PPL={r['ppl']:.4f} ({r['tokens']} tokens)")


if __name__ == "__main__":
    main()
