#!/usr/bin/env -S python -u
"""K-DTCBench 보고서용: 세 모델(BF16 / GPTQ 4-bit / FOEM 3-bit) 전체 240문항에 대해
A~D 확률분포를 수집해 JSON으로 저장한다 (보고서 5절 예시 선정 및 표 작성용)."""
import gc
import json

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoProcessor, AutoConfig

BASE_PATH = "/root/.cache/huggingface/hub/models--mistralai--Ministral-3-3B-Instruct-2512-BF16/snapshots/ecc3ba8b43a45610e709327c049d24b009bfec88"
GPTQ_PATH = "/workspace/LLM-VLM-in-Jetson/FOEM/Ministral-3-3B-Instruct-2512-BF16_gptq_4bit"
FOEM_PATH = "/workspace/LLM-VLM-in-Jetson/FOEM/Ministral-3-3B-Instruct-2512-BF16_foem_3bit"
OUT_PATH = "/workspace/LLM-VLM-in-Jetson/FOEM/logs/kdtcbench_predictions.json"

LETTERS = ["A", "B", "C", "D"]
CHOICE_KEYS = ["choice_a", "choice_b", "choice_c", "choice_d"]


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
    return hf, None


def load_quant(path):
    from gptqmodel import GPTQModel
    last_err = None
    for kwargs in (
        {"attn_implementation": "sdpa"},
        {"attn_implementation": "sdpa", "backend": "gptq_triton"},
        {"attn_implementation": "eager", "backend": "gptq_triton"},
    ):
        try:
            qm = GPTQModel.load(path, **kwargs)
            break
        except Exception as e:
            last_err = e
    else:
        raise last_err
    hf = getattr(qm, "model", qm)
    hf.eval()
    return hf, qm


def unload(hf, extra=None):
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


def _build_prompt(ex: dict) -> str:
    options = ", ".join(f"{l}: {ex[k]}" for l, k in zip(LETTERS, CHOICE_KEYS))
    return (
        f"{ex['question']}\nOptions: {options}\n\n"
        "주어진 선택지 중 해당 옵션의 문자로 바로 답하세요."
    )


@torch.inference_mode()
def predict_one(hf, processor, choice_ids, ex):
    messages = [{"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": _build_prompt(ex)},
    ]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(images=[ex["image"]], text=text, return_tensors="pt")
    dev = next(hf.parameters()).device
    inputs = {k: v.to(dev) for k, v in inputs.items()}

    logits = hf(**inputs).logits[0, -1, :]
    probs = F.softmax(torch.stack([logits[i] for i in choice_ids]).float(), dim=0)
    pred = int(probs.argmax())
    return {
        "pred": LETTERS[pred],
        "probs": {l: float(probs[i]) * 100 for i, l in enumerate(LETTERS)},
        "correct": LETTERS[pred] == ex["answer"],
    }


def run_model(label, hf, processor, dataset):
    choice_ids = [processor.tokenizer.encode(l, add_special_tokens=False)[0] for l in LETTERS]
    results = []
    for i, ex in enumerate(dataset):
        r = predict_one(hf, processor, choice_ids, ex)
        r.update({"index": ex["index"], "category": ex["category"], "answer": ex["answer"]})
        results.append(r)
        if (i + 1) % 40 == 0:
            print(f"  [{label}] {i+1}/{len(dataset)}")
    acc = sum(r["correct"] for r in results) / len(results)
    print(f"[{label}] accuracy = {acc:.2%}")
    return results


def main():
    processor = AutoProcessor.from_pretrained(BASE_PATH)
    dataset = load_dataset("NCSOFT/K-DTCBench", split="test")

    all_results = {}

    print("[load] BF16 베이스 모델")
    hf, extra = load_base()
    all_results["bf16"] = run_model("BF16", hf, processor, dataset)
    unload(hf, extra)

    print("[load] GPTQ 4-bit")
    hf, extra = load_quant(GPTQ_PATH)
    all_results["gptq_4bit"] = run_model("GPTQ", hf, processor, dataset)
    unload(hf, extra)

    print("[load] FOEM 3-bit")
    hf, extra = load_quant(FOEM_PATH)
    all_results["foem_3bit"] = run_model("FOEM", hf, processor, dataset)
    unload(hf, extra)

    # 질문/선택지 텍스트도 함께 저장 (이미지는 제외)
    meta = {
        ex["index"]: {
            "question": ex["question"],
            "choices": {l: ex[k] for l, k in zip(LETTERS, CHOICE_KEYS)},
            "answer": ex["answer"],
            "category": ex["category"],
        }
        for ex in dataset
    }

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "results": all_results}, f, ensure_ascii=False, indent=2)
    print(f"[done] saved -> {OUT_PATH}")


if __name__ == "__main__":
    main()
