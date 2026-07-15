#!/usr/bin/env -S python -u
"""보고서 섹션 5용: BF16 베이스 모델로 4개 예시 문항의 A/B/C/D 확률분포 수집."""
import os
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer, AutoConfig

BASE_PATH = "/root/.cache/huggingface/hub/models--mistralai--Ministral-3-3B-Instruct-2512-BF16/snapshots/ecc3ba8b43a45610e709327c049d24b009bfec88"
CHOICE_LETTERS = ["A", "B", "C", "D"]
SUBJECTS = ["Korean-History", "Math", "Computer-Science", "Marketing"]


def load_base_model():
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


def _format_choices(ex):
    return "\n".join(f"{l}. {ex[l]}" for l in CHOICE_LETTERS)


def _format_q(ex, with_answer):
    text = f"{ex['question']}\n{_format_choices(ex)}\n정답:"
    if with_answer:
        text += f" {CHOICE_LETTERS[ex['answer'] - 1]}"
    return text


def build_prefix(subject):
    dev = load_dataset("HAERAE-HUB/KMMLU", subject, split="dev")
    return "\n\n".join(_format_q(ex, True) for ex in dev.select(range(5))) + "\n\n"


@torch.inference_mode()
def predict(hf, tok, subject):
    prefix = build_prefix(subject)
    test = load_dataset("HAERAE-HUB/KMMLU", subject, split="test")
    ex = test[0]

    prompt = prefix + _format_q(ex, False)
    input_ids = tok(prompt, return_tensors="pt").input_ids.to(next(hf.parameters()).device)
    logits = hf(input_ids=input_ids).logits[0, -1, :]

    choice_ids = [tok.encode(f" {l}", add_special_tokens=False)[0] for l in CHOICE_LETTERS]
    choice_logits = torch.stack([logits[i] for i in choice_ids])
    probs = F.softmax(choice_logits.float(), dim=0)
    pred_idx = int(probs.argmax())

    return {
        "subject": subject,
        "question": ex["question"],
        "choices": {l: ex[l] for l in CHOICE_LETTERS},
        "answer": CHOICE_LETTERS[ex["answer"] - 1],
        "pred": CHOICE_LETTERS[pred_idx],
        "probs": {l: float(probs[i]) * 100 for i, l in enumerate(CHOICE_LETTERS)},
        "correct": pred_idx == ex["answer"] - 1,
    }


def main():
    tok = AutoTokenizer.from_pretrained(BASE_PATH)
    hf = load_base_model()

    results = []
    for subj in SUBJECTS:
        r = predict(hf, tok, subj)
        results.append(r)
        mark = "✓" if r["correct"] else "✗"
        print(f"[{subj}] pred={r['pred']} answer={r['answer']} {mark}")
        for l in CHOICE_LETTERS:
            print(f"  {l}: {r['probs'][l]:.1f}%")

    # 마크다운 행 출력 (보고서 붙여넣기용)
    print("\n--- 마크다운 행 (BF16 추가분) ---")
    for r in results:
        mark = "✓" if r["correct"] else "✗"
        ps = r["probs"]
        print(f"| BF16 원본 | **{r['pred']}** | {ps['A']:.1f}% | {ps['B']:.1f}% | {ps['C']:.1f}% | {ps['D']:.1f}% | {mark} |")


if __name__ == "__main__":
    main()
