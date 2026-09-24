"""
awq/export.py

AWQ quantized 모델을 vLLM이 로드할 수 있는 포맷으로 저장합니다.

vLLM AWQ 포맷 요구사항:
  - config.json에 "quantization_config": {"quant_type": "awq", "w_bit": 4, "group_size": 128}
  - safetensors에 각 Linear 레이어의 qweight / scales / zeros 포함
  - tokenizer 파일 동일하게 복사
"""

import json
import shutil
import torch
import yaml
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from safetensors.torch import save_file


def load_config(config_path: str = "../configs/config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def build_awq_config_json(base_config: dict, awq_cfg: dict) -> dict:
    """
    원본 모델의 config.json에 AWQ quantization 설정을 추가합니다.

    Args:
        base_config: 원본 모델의 config.json 딕셔너리
        awq_cfg    : config.yaml의 awq 섹션

    Returns:
        vLLM AWQ 포맷에 맞는 config 딕셔너리
    """
    config = base_config.copy()
    config["quantization_config"] = {
        "quant_type": "awq",
        "quant_method": "awq",
        "bits": awq_cfg["w_bit"],
        "w_bit": awq_cfg["w_bit"],
        "group_size": awq_cfg["group_size"],
        "zero_point": awq_cfg["zero_point"],
        "version": "GEMM",
    }
    return config


def export_awq_model(
    model: torch.nn.Module,
    quant_results: dict,          # quantize.py의 quantize_model 결과
    tokenizer,
    output_dir: str,
    config: dict,
) -> None:
    """
    AWQ quantized 모델을 output_dir에 저장합니다.

    저장 구조:
        output_dir/
          ├── config.json          (AWQ quantization_config 포함)
          ├── model.safetensors    (qweight, scales, zeros + 나머지 FP 파라미터)
          ├── tokenizer.json
          ├── tokenizer_config.json
          └── special_tokens_map.json

    Args:
        model        : AWQ가 적용된 모델
        quant_results: {layer_name: {"qweight", "scales", "zeros", "best_scale"}}
        tokenizer    : HuggingFace tokenizer
        output_dir   : 저장 경로
        config       : config.yaml 설정

    # TODO: 아래 구현부를 완성하세요.
    #
    # 단계:
    #   1. output_dir 생성
    #   2. 원본 config.json 로드 → build_awq_config_json()으로 AWQ 설정 추가 → 저장
    #   3. state_dict 순회:
    #        - quant_results에 있는 레이어 → qweight/scales/zeros로 교체
    #        - 나머지 파라미터 → 그대로 포함
    #   4. save_file()로 safetensors 저장
    #   5. tokenizer 저장 (tokenizer.save_pretrained)
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    awq_cfg = config["awq"]

    # 원본 config.json 로드 및 AWQ 설정 추가
    base_config = model.config.to_dict()
    awq_config = build_awq_config_json(base_config, awq_cfg)
    with open(output_path / "config.json", "w") as f:
        json.dump(awq_config, f, indent=2, ensure_ascii=False)

    # state_dict 구성: 양자화된 레이어는 qweight/qzeros/scales로 교체
    from .pack import pack_int4_weight, AWQ_PACK_ORDER

    state_dict = {}
    quant_layer_prefixes = set()
    w_bit = awq_cfg["w_bit"]
    for layer_name, qr in quant_results.items():
        quant_layer_prefixes.add(layer_name)
        state_dict[f"{layer_name}.qweight"] = qr["qweight"]
        # AWQ 표준: scales는 [n_groups, out_features] (전치)
        state_dict[f"{layer_name}.scales"] = qr["scales"].T.contiguous()
        if qr["zeros"] is not None:
            # AWQ 표준: zeros를 전치 → [n_groups, out_features] → out_features 방향으로 패킹
            zeros_int = qr["zeros"].cpu().to(torch.int32).T.contiguous()  # [n_groups, out_features]
            values_per_int32 = 32 // w_bit
            n_groups, out_f = zeros_int.shape
            if out_f % values_per_int32 == 0:
                zeros_reshaped = zeros_int.reshape(n_groups, out_f // values_per_int32, values_per_int32)
                qzeros = torch.zeros(n_groups, out_f // values_per_int32, dtype=torch.int32)
                for i in range(values_per_int32):
                    qzeros |= (zeros_reshaped[:, :, AWQ_PACK_ORDER[i]] << (i * w_bit))
                state_dict[f"{layer_name}.qzeros"] = qzeros
            else:
                state_dict[f"{layer_name}.qzeros"] = zeros_int

    # 양자화되지 않은 파라미터(embedding, norm 등)는 그대로 포함
    for name, param in model.named_parameters():
        layer_name = name.rsplit(".", 1)[0]  # "model.layers.0.self_attn.q_proj.weight" → "model.layers.0.self_attn.q_proj"
        if layer_name not in quant_layer_prefixes:
            state_dict[name] = param.data

    save_file(state_dict, str(output_path / "model.safetensors"))

    # tokenizer 저장
    tokenizer.save_pretrained(output_dir)

    print(f"AWQ 모델 저장 완료: {output_dir}")


def run_autoawq_baseline(config: dict) -> None:
    """
    AutoAWQ 라이브러리를 사용해 베이스라인 AWQ 모델을 생성합니다.
    직접 구현 결과와 비교하는 기준값으로 사용합니다.

    필요 패키지: pip install autoawq
    """
    from awq import AutoAWQForCausalLM

    model_name = config["paths"]["baseline_model"]
    output_dir = config["paths"]["awq_autoawq_output"]
    awq_cfg = config["awq"]

    quant_config = {
        "w_bit": awq_cfg["w_bit"],
        "q_group_size": awq_cfg["group_size"],
        "zero_point": awq_cfg["zero_point"],
        "version": "GEMM",
    }

    print(f"Loading model: {model_name}")
    model = AutoAWQForCausalLM.from_pretrained(model_name, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    print("Running AutoAWQ quantization (baseline)...")
    model.quantize(tokenizer, quant_config=quant_config)

    print(f"Saving AutoAWQ baseline to: {output_dir}")
    model.save_quantized(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("Done.")


def main():
    config = load_config()

    print("=== Generating AutoAWQ Baseline ===")
    run_autoawq_baseline(config)

    print("\nAutoAWQ 베이스라인 생성 완료.")
    print(f"저장 위치: {config['paths']['awq_autoawq_output']}")
    print("\n직접 구현 AWQ 내보내기는 export_awq_model()을 구현한 뒤 실행하세요.")


if __name__ == "__main__":
    main()
