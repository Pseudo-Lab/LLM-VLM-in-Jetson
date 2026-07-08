#!/usr/bin/env -S python -u
"""KMMLU (Korean MMLU) 정확도 평가 (양자화 모델 vs fp16 원본).

eval_ppl.py 와 동일한 이유로 lm-eval 대신 log-likelihood 기반 4지선다 채점을
직접 구현한다: 5-shot 프롬프트 뒤에 " A"/" B"/" C"/" D" 후보 토큰의 logit을
비교해 argmax 선택 (lm-eval-harness MMLU/KMMLU 태스크와 동일한 방식).
generate() 호출이 없어 멀티모달 래퍼와도 PPL 스크립트처럼 호환된다.

HAERAE-HUB/KMMLU 는 단일 "All" config 가 없고 45개 과목별 config 만 존재하므로,
전체 평가는 45개를 순회하며 누적한다.

  # 양자화 모델 (GPTQModel 포맷), 전체 45개 과목, 5-shot
  python eval_kmmlu.py --model quantized/Ministral-3-3B_gptq_4bit --quant
  # 과목 1개만 빠르게 확인
  python eval_kmmlu.py --model mistralai/Ministral-3-3B-Instruct-2512-BF16 \
      --subset Korean-History --limit-per-subject 20 --shots 5
"""
import argparse
import os
import time

import torch
from datasets import load_dataset
from transformers import AutoTokenizer

KMMLU_SUBJECTS = [
    "Accounting", "Agricultural-Sciences", "Aviation-Engineering-and-Maintenance",
    "Biology", "Chemical-Engineering", "Chemistry", "Civil-Engineering",
    "Computer-Science", "Construction", "Criminal-Law", "Ecology", "Economics",
    "Education", "Electrical-Engineering", "Electronics-Engineering",
    "Energy-Management", "Environmental-Science", "Fashion", "Food-Processing",
    "Gas-Technology-and-Engineering", "Geomatics", "Health", "Industrial-Engineer",
    "Information-Technology", "Interior-Architecture-and-Design", "Law",
    "Machine-Design-and-Manufacturing", "Management", "Maritime-Engineering",
    "Marketing", "Materials-Engineering", "Mechanical-Engineering",
    "Nondestructive-Testing", "Patent", "Political-Science-and-Sociology",
    "Psychology", "Public-Safety", "Railway-and-Automotive-Engineering",
    "Real-Estate", "Refrigerating-Machinery", "Social-Welfare", "Taxation",
    "Telecommunications-and-Wireless-Technology", "Korean-History", "Math",
]

CHOICE_LETTERS = ["A", "B", "C", "D"]


def load_model(path: str, quant: bool):
    if quant:
        from gptqmodel import GPTQModel
        m = GPTQModel.load(path, attn_implementation="sdpa")
        hf = getattr(m, "model", m)
        return hf, m
    else:
        from huggingface_hub import snapshot_download
        from transformers import AutoConfig
        p = snapshot_download(path, local_files_only=True) if "/" in path and not os.path.isdir(path) else path
        cfg = AutoConfig.from_pretrained(p)
        # BF16 베이스 모델은 tie_word_embeddings=True 상태로 저장돼 lm_head.weight가 체크포인트에 없음.
        # False로 강제하면 lm_head가 랜덤 초기화되므로 원본 설정 그대로 유지.
        import transformers as _t
        cls = getattr(_t, cfg.architectures[0])
        hf = cls.from_pretrained(
            p, config=cfg, dtype=torch.bfloat16, device_map="auto", attn_implementation="sdpa"
        )
        hf.tie_weights()  # 명시적으로 lm_head ← embed_tokens 결합 확인
        return hf, hf


def _format_choices(example: dict) -> str:
    return "\n".join(f"{letter}. {example[letter]}" for letter in CHOICE_LETTERS)


def _format_question(example: dict, with_answer: bool) -> str:
    text = f"{example['question']}\n{_format_choices(example)}\n정답:"
    if with_answer:
        text += f" {CHOICE_LETTERS[example['answer'] - 1]}"
    return text


def build_few_shot_prefix(subject: str, shots: int) -> str:
    if shots <= 0:
        return ""
    dev = load_dataset("HAERAE-HUB/KMMLU", subject, split="dev")
    examples = dev.select(range(min(shots, len(dev))))
    return "\n\n".join(_format_question(ex, with_answer=True) for ex in examples) + "\n\n"


def _choice_token_ids(tokenizer) -> list[int]:
    return [tokenizer.encode(f" {letter}", add_special_tokens=False)[0] for letter in CHOICE_LETTERS]


@torch.inference_mode()
def evaluate_subject(hf_model, tokenizer, subject: str, shots: int, limit: int | None = None) -> dict:
    prefix = build_few_shot_prefix(subject, shots)
    choice_ids = _choice_token_ids(tokenizer)
    dev = next(hf_model.parameters()).device

    test = load_dataset("HAERAE-HUB/KMMLU", subject, split="test")
    if limit is not None:
        test = test.select(range(min(limit, len(test))))

    correct = 0
    for ex in test:
        prompt = prefix + _format_question(ex, with_answer=False)
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(dev)
        logits = hf_model(input_ids=input_ids).logits[0, -1, :]
        pred = int(torch.stack([logits[i] for i in choice_ids]).argmax())
        if pred == ex["answer"] - 1:
            correct += 1

    return {"correct": correct, "total": len(test)}


def evaluate_all(hf_model, tokenizer, subjects: list[str], shots: int, limit: int | None = None) -> dict:
    per_subject = {}
    for i, subject in enumerate(subjects, 1):
        r = evaluate_subject(hf_model, tokenizer, subject, shots, limit)
        acc = r["correct"] / r["total"] if r["total"] else 0.0
        per_subject[subject] = r
        print(f"[kmmlu] {subject}: {acc:.2%} ({r['correct']}/{r['total']})  [{i}/{len(subjects)}]")

    total_correct = sum(r["correct"] for r in per_subject.values())
    total_count = sum(r["total"] for r in per_subject.values())
    accuracy = total_correct / total_count if total_count else 0.0
    subject_accs = [r["correct"] / r["total"] for r in per_subject.values() if r["total"]]
    macro_accuracy = sum(subject_accs) / len(subject_accs) if subject_accs else 0.0

    return {
        "accuracy": accuracy,
        "macro_accuracy": macro_accuracy,
        "per_subject": per_subject,
        "n_questions": total_count,
        "shots": shots,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--quant", action="store_true")
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--subset", nargs="+", default=None,
                    help="평가할 과목 목록 (기본: KMMLU 45개 과목 전체)")
    ap.add_argument("--shots", type=int, default=5)
    ap.add_argument("--limit-per-subject", type=int, default=None)
    args = ap.parse_args()

    subjects = args.subset or KMMLU_SUBJECTS
    tok = AutoTokenizer.from_pretrained(args.tokenizer or args.model)
    hf, _ = load_model(args.model, args.quant)
    hf.eval()

    t0 = time.time()
    result = evaluate_all(hf, tok, subjects, args.shots, args.limit_per_subject)
    elapsed = time.time() - t0

    print(f"RESULT\t{args.model}\tKMMLU_Acc\t{result['accuracy']:.4f}")
    print(f"[kmmlu] micro={result['accuracy']:.2%}  macro={result['macro_accuracy']:.2%}  "
          f"n={result['n_questions']}  elapsed={elapsed/60:.1f}분")


if __name__ == "__main__":
    main()
