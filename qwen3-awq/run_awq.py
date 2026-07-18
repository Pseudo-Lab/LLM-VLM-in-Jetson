"""
run_awq.py

AWQ 양자화 CLI 실행 스크립트.

사용법:
  python run_awq.py --model Qwen/Qwen3-4B --calib-data pileval
  python run_awq.py --model meta-llama/Llama-3.2-3B --calib-data wikitext2
  python run_awq.py --model mistralai/Mistral-7B-v0.3 --calib-data c4
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import importlib.util


def load_module(name, path):
    """autoawq 패키지 충돌 방지를 위한 명시적 모듈 로드."""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# autoawq가 설치된 환경에서 awq/ 폴더와 이름 충돌 방지
awq_dir = Path(__file__).parent / "awq"
load_module("awq.calibration", awq_dir / "calibration.py")
load_module("awq.pack", awq_dir / "pack.py")
load_module("awq.quantize", awq_dir / "quantize.py")
load_module("awq.export", awq_dir / "export.py")
pipeline = load_module("awq.pipeline", awq_dir / "pipeline.py")

AWQQuantizer = pipeline.AWQQuantizer


def main():
    parser = argparse.ArgumentParser(description="AWQ Quantization Pipeline")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-4B",
                        help="HuggingFace 모델 ID 또는 로컬 경로")
    parser.add_argument("--calib-data", type=str, default="pileval",
                        choices=["pileval", "wikitext2", "kowikitext", "c4"],
                        help="Calibration 데이터셋")
    parser.add_argument("--n-samples", type=int, default=128,
                        help="Calibration 샘플 수")
    parser.add_argument("--seq-len", type=int, default=512,
                        help="Calibration 시퀀스 길이")
    parser.add_argument("--w-bit", type=int, default=4,
                        help="양자화 비트 수")
    parser.add_argument("--group-size", type=int, default=128,
                        help="그룹 양자화 크기")
    parser.add_argument("--clip-search", action="store_true",
                        help="그룹별 weight clipping 탐색 활성화 (outlier로 인한 해상도 낭비 방지)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="출력 디렉토리 (기본: ./outputs/<model>-awq-<calib>)")
    parser.add_argument("--skip-layers", type=str, nargs="*", default=None,
                        help="양자화에서 제외할 레이어 이름 (기본: lm_head)")
    args = parser.parse_args()

    skip_layers = set(args.skip_layers) if args.skip_layers else None

    quantizer = AWQQuantizer(
        model_name=args.model,
        w_bit=args.w_bit,
        group_size=args.group_size,
        clip_search=args.clip_search,
        skip_layers=skip_layers,
    )

    quantizer.quantize(
        calib_data=args.calib_data,
        output_dir=args.output_dir,
        n_samples=args.n_samples,
        seq_len=args.seq_len,
    )


if __name__ == "__main__":
    main()
