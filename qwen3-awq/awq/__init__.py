"""
awq — Activation-aware Weight Quantization

HuggingFace CausalLM 모델에 대한 AWQ INT4 양자화 파이프라인.

사용법:
    from awq import AWQQuantizer

    quantizer = AWQQuantizer(
        model_name="Qwen/Qwen3-4B",
        w_bit=4,
        group_size=128,
    )
    quantizer.quantize(calib_data="pileval", output_dir="./outputs/qwen3-4b-awq")
"""

from .pipeline import AWQQuantizer

__all__ = ["AWQQuantizer"]
