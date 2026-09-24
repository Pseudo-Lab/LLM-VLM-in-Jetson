"""양자화 전/후 + 엣지 참고선 비교표 생성 (데스크톱).

비교 설계(핵심):
  1) 순수 양자화 검증  : 11B fp16  vs  11B 4bit  ← 같은 모델이라 차이=양자화 효과
  2) 엣지 배포 참고선   : SmolVLM2-2.2B(무양자화) ← 별개 참고용, 양자화 기준 아님

fp16 11B 는 22GB 라 Jetson 엔 안 올라가지만 정확도는 하드웨어 무관이므로
데스크톱(3080Ti + CPU 오프로드)에서 측정한다. Jetson 측정은 scripts/jetson/ 참고.

실행:
  python src/evaluate.py
비교 대상/데이터셋은 configs/gptq_config.yaml 의 evaluate: 섹션에서 조정.
"""
import json

from datasets import load_dataset

from config import load_config, hf_token, resolve
from benchmark import load_model, run_benchmark


def _question_of(row) -> str:
    return row.get("question") or "Describe this image."


def _answer_of(row):
    return row.get("answers") or row.get("answer")  # 없으면 None → 정확도 미산출


def load_eval_samples(ecfg: dict) -> list[dict]:
    ds = load_dataset(ecfg["dataset"], split=ecfg.get("split", "validation"), streaming=True)
    samples = []
    for row in ds:
        if len(samples) >= ecfg["num_samples"]:
            break
        image = row.get("image")
        if image is None:
            continue
        samples.append(
            {"image": image, "question": _question_of(row), "answer": _answer_of(row)}
        )
    print(f"[evaluate] 평가 샘플 {len(samples)}개 (dataset={ecfg['dataset']})")
    return samples


def _print_table(results: list[dict]) -> None:
    cols = ["name", "peak_vram_gb", "latency_s", "tokens_per_s", "accuracy"]
    widths = {c: max(len(c), *(len(str(r.get(c, ""))) for r in results)) for c in cols}
    line = " | ".join(c.ljust(widths[c]) for c in cols)
    print("\n" + line)
    print("-" * len(line))
    for r in results:
        print(" | ".join(str(r.get(c, "")).ljust(widths[c]) for c in cols))


def main() -> None:
    cfg = load_config()
    ecfg = cfg["evaluate"]
    token = hf_token()

    samples = load_eval_samples(ecfg)

    results = []
    for entry in ecfg["models"]:
        # 로컬 경로면 절대경로로, HF repo id 면 그대로
        path = entry["path"]
        p = resolve(path)
        path = str(p) if p.exists() else path

        print(f"\n[evaluate] '{entry['name']}' ({entry['kind']}) 로딩…")
        model, processor = load_model(entry["kind"], path, token=token)
        res = run_benchmark(
            model, processor, samples, entry["name"],
            max_new_tokens=ecfg.get("max_new_tokens", 64),
        )
        results.append(res.as_dict())

        # 다음 모델 위해 메모리 해제
        del model
        import gc, torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    _print_table(results)

    out = resolve("results/eval_comparison.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[evaluate] 결과 저장 → {out}")


if __name__ == "__main__":
    main()
