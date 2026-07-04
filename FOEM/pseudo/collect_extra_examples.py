#!/usr/bin/env -S python -u
"""보고서 섹션 5 추가 예시용: Education / Patent / Law / Information-Technology
세 모델(BF16 / GPTQ 4-bit / FOEM 3-bit) 순서로 로드해 A~D 확률분포 수집."""
import gc
import os
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer, AutoConfig

BASE_PATH  = "/root/.cache/huggingface/hub/models--mistralai--Ministral-3-3B-Instruct-2512-BF16/snapshots/ecc3ba8b43a45610e709327c049d24b009bfec88"
GPTQ_PATH  = "/workspace/LLM-VLM-in-Jetson/Ministral-3-3B-Instruct-2512-BF16_gptq_4bit"
FOEM_PATH  = "/workspace/LLM-VLM-in-Jetson/Ministral-3-3B-Instruct-2512-BF16_foem_3bit"

SUBJECTS   = ["Education", "Patent", "Law", "Information-Technology"]
LETTERS    = ["A", "B", "C", "D"]


# ── 모델 로더 ─────────────────────────────────────────────────────────────────

def load_base():
    cfg = AutoConfig.from_pretrained(BASE_PATH)
    import transformers as _t
    cls = getattr(_t, cfg.architectures[0])
    hf = cls.from_pretrained(
        BASE_PATH, config=cfg, dtype=torch.bfloat16,
        device_map="auto", attn_implementation="sdpa"
    )
    hf.tie_weights()
    hf.eval()
    return hf

def load_quant(path):
    from gptqmodel import GPTQModel
    try:
        qm = GPTQModel.load(path, attn_implementation="eager")
    except Exception:
        qm = GPTQModel.load(path, attn_implementation="eager", backend="gptq_triton")
    hf = getattr(qm, "model", qm)
    hf.eval()
    return hf, qm

def unload(hf, extra=None):
    # GPU → CPU 이동 후 삭제해야 VRAM이 즉시 반환됨
    try:
        hf.cpu()
    except Exception:
        pass
    del hf
    if extra is not None:
        del extra
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


# ── 추론 ─────────────────────────────────────────────────────────────────────

def _fmt(ex, with_ans):
    choices = "\n".join(f"{l}. {ex[l]}" for l in LETTERS)
    text = f"{ex['question']}\n{choices}\n정답:"
    if with_ans:
        text += f" {LETTERS[ex['answer'] - 1]}"
    return text

def build_prefix(subject):
    dev = load_dataset("HAERAE-HUB/KMMLU", subject, split="dev")
    return "\n\n".join(_fmt(ex, True) for ex in dev.select(range(5))) + "\n\n"

@torch.inference_mode()
def predict_one(hf, tok, subject):
    prefix = build_prefix(subject)
    ex = load_dataset("HAERAE-HUB/KMMLU", subject, split="test")[0]
    prompt = prefix + _fmt(ex, False)
    dev = next(hf.parameters()).device
    ids = tok(prompt, return_tensors="pt").input_ids.to(dev)
    logits = hf(input_ids=ids).logits[0, -1, :]
    cids = [tok.encode(f" {l}", add_special_tokens=False)[0] for l in LETTERS]
    probs = F.softmax(torch.stack([logits[i] for i in cids]).float(), dim=0)
    pred  = int(probs.argmax())
    return {
        "subject":  subject,
        "question": ex["question"],
        "choices":  {l: ex[l] for l in LETTERS},
        "answer":   LETTERS[ex["answer"] - 1],
        "pred":     LETTERS[pred],
        "probs":    {l: float(probs[i]) * 100 for i, l in enumerate(LETTERS)},
        "correct":  pred == ex["answer"] - 1,
    }


# ── 메인 ─────────────────────────────────────────────────────────────────────

def run_model(label, hf, tok):
    print(f"\n=== {label} ===")
    results = {}
    for s in SUBJECTS:
        r = predict_one(hf, tok, s)
        results[s] = r
        mark = "✓" if r["correct"] else "✗"
        ps = r["probs"]
        print(f"  [{s}] pred={r['pred']} ans={r['answer']} {mark}  "
              f"A:{ps['A']:.1f}% B:{ps['B']:.1f}% C:{ps['C']:.1f}% D:{ps['D']:.1f}%")
    return results


def main():
    tok = AutoTokenizer.from_pretrained(BASE_PATH)

    # BF16
    print("[load] BF16 베이스 모델")
    hf = load_base()
    base_res = run_model("BF16 원본", hf, tok)
    unload(hf)

    # GPTQ
    print("[load] GPTQ 4-bit")
    hf, qm = load_quant(GPTQ_PATH)
    gptq_res = run_model("GPTQ 4-bit", hf, tok)
    unload(hf, qm)

    # FOEM
    print("[load] FOEM 3-bit")
    hf, qm = load_quant(FOEM_PATH)
    foem_res = run_model("FOEM 3-bit", hf, tok)
    unload(hf, qm)

    # ── 마크다운 출력 ──────────────────────────────────────────────────────────
    print("\n\n" + "="*60)
    print("마크다운 섹션 (보고서 붙여넣기용)")
    print("="*60)

    labels = [("BF16 원본", base_res), ("GPTQ 4-bit", gptq_res), ("FOEM 3-bit", foem_res)]

    for s in SUBJECTS:
        b = base_res[s]
        print(f"\n### {s}")
        print(f"질문: {b['question']}")
        for l in LETTERS:
            print(f"  {l}. {b['choices'][l]}")
        print(f"정답: {b['answer']}")
        print()
        print("| 모델 | 예측 | A | B | C | D | 정오 |")
        print("|---|---|---:|---:|---:|---:|---|")
        for label, res in labels:
            r = res[s]
            ps = r["probs"]
            mark = "✓" if r["correct"] else "✗"
            # 최고확률에 ** 강조
            max_l = max(LETTERS, key=lambda l: ps[l])
            cells = {l: f"**{ps[l]:.1f}%**" if l == max_l else f"{ps[l]:.1f}%" for l in LETTERS}
            print(f"| {label} | **{r['pred']}** | {cells['A']} | {cells['B']} | {cells['C']} | {cells['D']} | {mark} |")


if __name__ == "__main__":
    main()
