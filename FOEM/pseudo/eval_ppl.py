#!/usr/bin/env python
"""WikiText-2 perplexity 평가 (양자화 모델 vs fp16 원본).

lm-eval 의 모델 래퍼가 mistral3(멀티모달) 양자화 모델과 충돌할 수 있어,
표준 sliding-window PPL 을 직접 계산한다. 지표 자체는 FOEM 논문이 보고하는
WikiText perplexity 와 동일.

  # 양자화 모델 (GPTQModel 포맷)
  python eval_ppl.py --model quantized/MistralSmall3.1-24B-gptq-4bit --quant
  # fp16 원본
  python eval_ppl.py --model mistralai/Mistral-Small-3.1-24B-Instruct-2503
"""
import argparse
import os

import torch
from datasets import load_dataset
from transformers import AutoTokenizer


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
        # 멀티모달 등 비-CausalLM 도 처리: 아키텍처에서 클래스를 그대로 import.
        # mistral3 처럼 lm_head 와 embed_tokens 가 양쪽 다 저장돼있는데
        # tie_word_embeddings 기본값이 True 인 경우 lm_head 가 무시되어
        # PPL 이 망가지므로 명시적으로 False 로 강제.
        cfg = AutoConfig.from_pretrained(p)
        cfg.tie_word_embeddings = False
        if hasattr(cfg, "text_config"):
            cfg.text_config.tie_word_embeddings = False
        import transformers as _t
        cls = getattr(_t, cfg.architectures[0])
        hf = cls.from_pretrained(
            p, config=cfg, dtype=torch.bfloat16, device_map="auto", attn_implementation="sdpa"
        )
        return hf, hf


@torch.inference_mode()
def ppl(hf_model, tokenizer, seqlen=2048):
    test = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    enc = tokenizer("\n\n".join(test["text"]), return_tensors="pt").input_ids
    dev = next(hf_model.parameters()).device
    n = enc.size(1) // seqlen
    nlls = []
    for i in range(n):
        batch = enc[:, i * seqlen:(i + 1) * seqlen].to(dev)
        # 멀티모달 래퍼라도 input_ids 만 주면 language model 로짓이 나온다.
        logits = hf_model(input_ids=batch).logits
        shift_logits = logits[:, :-1, :].float()
        shift_labels = batch[:, 1:]
        loss = torch.nn.functional.cross_entropy(
            shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1)
        )
        if torch.isfinite(loss):
            nlls.append(loss * (seqlen - 1))
    if not nlls:
        return float("nan")
    return torch.exp(torch.stack(nlls).sum() / (len(nlls) * (seqlen - 1))).item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--quant", action="store_true")
    ap.add_argument("--seqlen", type=int, default=2048)
    ap.add_argument("--tokenizer", default="mistralai/Mistral-Small-3.1-24B-Instruct-2503")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    hf, _ = load_model(args.model, args.quant)
    hf.eval()
    p = ppl(hf, tok, args.seqlen)
    print(f"RESULT\t{args.model}\tWikiText2_PPL\t{p:.4f}")


if __name__ == "__main__":
    main()
