"""Jetson Orin Nano 온디바이스 추론 벤치.

데스크톱 evaluate.py 와 같은 측정 로직(src/benchmark.py)을 재사용해, Jetson 에서
실제로 올라가는 모델만 단독으로 잰다:
  - 11B GPTQ-4bit  (kind=gptq)        : 양자화 모델의 엣지 적재 가능성/속도
  - SmolVLM2-2.2B   (kind=smolvlm)     : 무양자화 엣지 참고선

fp16 11B 는 22GB 라 Jetson 에 안 올라가므로 여기서 측정하지 않는다(정확도는
데스크톱 evaluate.py 에서 측정).

사용:
  python scripts/jetson/bench_jetson.py --kind gptq    --path <양자화_모델_경로>
  python scripts/jetson/bench_jetson.py --kind smolvlm --path HuggingFaceTB/SmolVLM2-2.2B-Instruct

주의(Phase 2 미검증): Jetson(aarch64)에서 gptqmodel/mllama 런타임은 별도 검증 필요.
torch/transformers 는 Jetson 용 빌드를 써야 하며, gptq 커널이 안 맞으면
llama.cpp(GGUF) 경로로 대체하는 것을 README 참고.
"""
import argparse
import json
import sys
from pathlib import Path

# src/ 모듈 사용
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from benchmark import load_model, run_benchmark  # noqa: E402


def load_samples(dataset: str, split: str, n: int):
    from datasets import load_dataset

    ds = load_dataset(dataset, split=split, streaming=True)
    out = []
    for row in ds:
        if len(out) >= n:
            break
        if row.get("image") is None:
            continue
        out.append(
            {
                "image": row["image"],
                "question": row.get("question") or "Describe this image.",
                "answer": row.get("answers") or row.get("answer"),
            }
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", required=True, choices=["gptq", "smolvlm", "mllama_fp16"])
    ap.add_argument("--path", required=True, help="모델 경로 또는 HF repo id")
    ap.add_argument("--dataset", default="lmms-lab/textvqa")
    ap.add_argument("--split", default="validation")
    ap.add_argument("--num-samples", type=int, default=50)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    args = ap.parse_args()

    samples = load_samples(args.dataset, args.split, args.num_samples)
    print(f"[jetson] {len(samples)}개 샘플로 '{args.path}' ({args.kind}) 측정")

    model, processor = load_model(args.kind, args.path)
    res = run_benchmark(model, processor, samples, args.path, max_new_tokens=args.max_new_tokens)

    print(json.dumps(res.as_dict(), ensure_ascii=False, indent=2))

    out = ROOT / "results" / f"jetson_{args.kind}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res.as_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[jetson] 저장 → {out}")


if __name__ == "__main__":
    main()
