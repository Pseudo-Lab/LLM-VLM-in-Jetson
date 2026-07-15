#!/usr/bin/env -S python -u
"""vLLM으로 FOEM 4bit 베이스라인 모델을 서빙해 기본 한국어 질의 응답을 로그로 남긴다.

FOEM 3bit는 vLLM의 GPTQ 백엔드가 3-bit를 지원하지 않아 로드 자체가 불가능함이
확인됨 (`Unsupported quantization config: bits=3, sym=True`). 동일 베이스라인
설정(alpha=0.0, beta=0.2, group_size=128)으로 4-bit 재양자화해 대체."""
import json
import time

from vllm import LLM, SamplingParams

MODEL_PATH = "/workspace/LLM-VLM-in-Jetson/FOEM/Ministral-3-3B-Instruct-2512-BF16_foem_4bit"
# 양자화 디렉토리에는 프로세서/토크나이저 부속 설정(processor_config.json 등)이 없어
# vLLM 로딩 시 여러 호환성 에러가 남 (eval_kdtcbench.py 등 기존 스크립트와 동일하게
# 원본 BF16 스냅샷에서 토크나이저를 읽고, 가중치만 양자화 디렉토리에서 읽는다).
TOKENIZER_PATH = "/root/.cache/huggingface/hub/models--mistralai--Ministral-3-3B-Instruct-2512-BF16/snapshots/ecc3ba8b43a45610e709327c049d24b009bfec88"
LOG_PATH = "/workspace/LLM-VLM-in-Jetson/FOEM/logs/vllm_korean_smoketest.log"

PROMPTS = [
    "대한민국의 수도는 어디인가요?",
    "김치찌개를 맛있게 끓이는 방법을 3단계로 간단히 설명해줘.",
    "인공지능이 무엇인지 한 문단으로 설명해줘.",
    "1부터 10까지 더하면 얼마야?",
    "오늘 기분이 좋아. 짧은 한국어 시를 하나 지어줘.",
]


def main():
    t0 = time.time()
    print(f"[load] {MODEL_PATH}")
    llm = LLM(
        model=MODEL_PATH,
        quantization="gptq",
        dtype="bfloat16",
        gpu_memory_utilization=0.85,
        max_model_len=4096,
        enforce_eager=True,
        trust_remote_code=True,
        tokenizer_mode="mistral",
    )
    print(f"[load] done ({time.time()-t0:.1f}s)")

    sampling_params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=256)

    results = []
    for i, prompt in enumerate(PROMPTS, 1):
        messages = [{"role": "user", "content": prompt}]
        t1 = time.time()
        outputs = llm.chat([messages], sampling_params)
        elapsed = time.time() - t1
        text = outputs[0].outputs[0].text
        print(f"\n=== [{i}/{len(PROMPTS)}] ({elapsed:.1f}s) ===")
        print(f"Q: {prompt}")
        print(f"A: {text}")
        results.append({"question": prompt, "answer": text, "elapsed_sec": elapsed})

    with open(LOG_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "model": MODEL_PATH,
            "quantization": "FOEM 3bit baseline (group_size=128, alpha=0.0, beta=0.2)",
            "results": results,
        }, f, ensure_ascii=False, indent=2)
    print(f"\n[done] saved -> {LOG_PATH}")


if __name__ == "__main__":
    main()
