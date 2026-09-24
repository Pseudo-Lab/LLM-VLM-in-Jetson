#!/usr/bin/env -S python -u
"""serve_foem.py 서버에 기본 한국어 질의를 보내고 응답을 로그로 남긴다."""
import json
import time

import requests

BASE_URL = "http://127.0.0.1:8000"
LOG_PATH = "/workspace/LLM-VLM-in-Jetson/FOEM/logs/serve_foem_korean_queries.log"

PROMPTS = [
    "대한민국의 수도는 어디인가요?",
    "김치찌개를 맛있게 끓이는 방법을 3단계로 간단히 설명해줘.",
    "인공지능이 무엇인지 한 문단으로 설명해줘.",
    "1부터 10까지 더하면 얼마야?",
    "오늘 기분이 좋아. 짧은 한국어 시를 하나 지어줘.",
]


def main():
    r = requests.get(f"{BASE_URL}/health", timeout=10)
    print("[health]", r.json())

    results = []
    for i, prompt in enumerate(PROMPTS, 1):
        t0 = time.time()
        r = requests.post(f"{BASE_URL}/chat", json={"message": prompt}, timeout=120)
        r.raise_for_status()
        data = r.json()
        wall = time.time() - t0
        print(f"\n=== [{i}/{len(PROMPTS)}] server={data['elapsed_sec']:.1f}s wall={wall:.1f}s ===")
        print(f"Q: {prompt}")
        print(f"A: {data['response']}")
        results.append({
            "question": prompt,
            "answer": data["response"],
            "server_elapsed_sec": data["elapsed_sec"],
            "wall_elapsed_sec": wall,
        })

    with open(LOG_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "model": "FOEM 3bit baseline (group_size=128, alpha=0.0, beta=0.2)",
            "serving": "FastAPI + HF transformers generate() (model loaded once at startup)",
            "results": results,
        }, f, ensure_ascii=False, indent=2)
    print(f"\n[done] saved -> {LOG_PATH}")


if __name__ == "__main__":
    main()
