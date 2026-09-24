#!/usr/bin/env -S python -u
"""K-DTCBench (한국어 문서/표/차트 VQA) 정확도 평가 (양자화 모델 vs BF16 원본).

eval_kmmlu.py 와 동일한 이유로 generate() 대신 log-likelihood 기반 4지선다
채점을 직접 구현한다: 공식 프롬프트로 채팅 템플릿을 구성한 뒤 [/INST] 직후
"A"/"B"/"C"/"D" (공백 없음) 후보 토큰의 logit 을 비교해 argmax 선택.

K-DTCBench 는 NCSOFT/K-DTCBench (test 단일 split, 240문항: document/table/chart
각 80개, digital/handwritten 각 50%)이며 dev split 이 없어 zero-shot 평가만
가능하다 (공식 벤치마크 프로토콜과 동일).

  # 양자화 모델 (GPTQModel 포맷)
  python eval_kdtcbench.py --model /workspace/LLM-VLM-in-Jetson/FOEM/Ministral-3-3B-Instruct-2512-BF16_gptq_4bit --quant
  # BF16 원본
  python eval_kdtcbench.py --model mistralai/Ministral-3-3B-Instruct-2512-BF16
"""
import argparse
import os
import time

import torch
from datasets import load_dataset
from transformers import AutoProcessor

CATEGORIES = ["document", "table", "chart"]
CHOICE_LETTERS = ["A", "B", "C", "D"]
CHOICE_KEYS = ["choice_a", "choice_b", "choice_c", "choice_d"]


def load_model(path: str, quant: bool):
    if quant:
        from gptqmodel import GPTQModel
        # sdpa 필수: eager 어텐션은 고해상도 문서 이미지에서 O(N^2) 어텐션 행렬이 커 OOM.
        # 백엔드는 비트폭에 따라 다르게 실패한다 — 4bit 기본 backend(Marlin)는
        # sm_120(Blackwell)에서 bf16 커널 JIT 컴파일 실패, 3bit는 TritonV2Linear가
        # bits∈{2,4,8}만 지원. auto→gptq_triton→eager 순으로 순차 폴백.
        last_err = None
        for kwargs in (
            {"attn_implementation": "sdpa"},
            {"attn_implementation": "sdpa", "backend": "gptq_triton"},
            {"attn_implementation": "eager", "backend": "gptq_triton"},
        ):
            try:
                m = GPTQModel.load(path, **kwargs)
                break
            except Exception as e:
                last_err = e
        else:
            raise last_err
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


def _build_prompt(ex: dict) -> str:
    options = ", ".join(f"{l}: {ex[k]}" for l, k in zip(CHOICE_LETTERS, CHOICE_KEYS))
    return (
        f"{ex['question']}\nOptions: {options}\n\n"
        "주어진 선택지 중 해당 옵션의 문자로 바로 답하세요."
    )


def _choice_token_ids(processor) -> list[int]:
    # [/INST] 직후에는 공백 없이 바로 "A"/"B"/"C"/"D" 토큰이 온다 (" A" 등 공백 포함
    # 토큰과는 다른 id). 실제 forward 로 top-token 이 이 매핑과 일치함을 확인함.
    tok = processor.tokenizer
    return [tok.encode(letter, add_special_tokens=False)[0] for letter in CHOICE_LETTERS]


@torch.inference_mode()
def evaluate_one(hf_model, processor, choice_ids: list[int], ex: dict) -> dict:
    messages = [{"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": _build_prompt(ex)},
    ]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(images=[ex["image"]], text=text, return_tensors="pt")
    dev = next(hf_model.parameters()).device
    inputs = {k: v.to(dev) for k, v in inputs.items()}

    logits = hf_model(**inputs).logits[0, -1, :]
    pred = int(torch.stack([logits[i] for i in choice_ids]).argmax())
    answer_idx = CHOICE_LETTERS.index(ex["answer"])
    return {"pred": CHOICE_LETTERS[pred], "answer": ex["answer"], "correct": pred == answer_idx}


def evaluate_all(hf_model, processor, dataset, limit: int | None = None) -> dict:
    choice_ids = _choice_token_ids(processor)
    if limit is not None:
        dataset = dataset.select(range(min(limit, len(dataset))))

    per_category = {c: {"correct": 0, "total": 0} for c in CATEGORIES}
    for i, ex in enumerate(dataset, 1):
        r = evaluate_one(hf_model, processor, choice_ids, ex)
        cat = ex["category"]
        per_category[cat]["total"] += 1
        if r["correct"]:
            per_category[cat]["correct"] += 1
        if i % 20 == 0 or i == len(dataset):
            print(f"[kdtcbench] {i}/{len(dataset)} 완료 ({cat}: {r['pred']} vs {r['answer']})")

    total_correct = sum(v["correct"] for v in per_category.values())
    total_count = sum(v["total"] for v in per_category.values())
    accuracy = total_correct / total_count if total_count else 0.0
    cat_accs = [v["correct"] / v["total"] for v in per_category.values() if v["total"]]
    macro_accuracy = sum(cat_accs) / len(cat_accs) if cat_accs else 0.0

    for c in CATEGORIES:
        v = per_category[c]
        acc = v["correct"] / v["total"] if v["total"] else 0.0
        print(f"[kdtcbench] {c}: {acc:.2%} ({v['correct']}/{v['total']})")

    return {
        "accuracy": accuracy,
        "macro_accuracy": macro_accuracy,
        "per_category": per_category,
        "n_questions": total_count,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--quant", action="store_true")
    ap.add_argument("--processor", default=None)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    processor = AutoProcessor.from_pretrained(args.processor or args.model)
    hf, _ = load_model(args.model, args.quant)
    hf.eval()

    dataset = load_dataset("NCSOFT/K-DTCBench", split="test")

    t0 = time.time()
    result = evaluate_all(hf, processor, dataset, args.limit)
    elapsed = time.time() - t0

    print(f"RESULT\t{args.model}\tKDTCBench_Acc\t{result['accuracy']:.4f}")
    print(f"[kdtcbench] micro={result['accuracy']:.2%}  macro={result['macro_accuracy']:.2%}  "
          f"n={result['n_questions']}  elapsed={elapsed/60:.1f}분")


if __name__ == "__main__":
    main()
