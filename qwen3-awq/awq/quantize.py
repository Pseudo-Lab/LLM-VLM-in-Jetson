"""
awq/quantize.py

AWQ (Activation-aware Weight Quantization) 핵심 구현.

AWQ 알고리즘 흐름:
  1. calibration으로 각 레이어 입력의 채널별 activation scale (s) 수집
  2. salient channel 보호: weight에 역방향 스케일링 적용 → W' = W * diag(s)
  3. 입력에도 역스케일 보정:             X' = X * diag(1/s)
  4. W'를 INT4로 quantize
  5. inference 시 W_quant * diag(s) 형태로 복원

참고: Lin et al., "AWQ: Activation-aware Weight Quantization for LLM Compression
      and Acceleration", 2023.
"""

import torch
import torch.nn as nn
import yaml
from tqdm import tqdm
from typing import Optional


def load_config(config_path: str = "../configs/config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def _materialize_weight(linear: nn.Linear) -> torch.Tensor:
    """meta device에 있는 weight를 CPU로 가져옵니다."""
    w = linear.weight.data
    if w.device.type == "meta":
        return linear.weight.data.to("cpu").clone()
    return w.clone()


# ---------------------------------------------------------------------------
# Quantization 유틸
# ---------------------------------------------------------------------------

def pseudo_quantize_tensor(
    w: torch.Tensor,
    w_bit: int = 4,
    group_size: int = 128,
    zero_point: bool = True,
    clip_search: bool = False,
    clip_ratios: tuple = (1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7),
) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """
    Weight 텐서를 symmetric / asymmetric INT-N으로 pseudo-quantize합니다.
    (실제 INT 저장이 아닌, FP에서 round-trip을 거친 dequantized 값 반환)

    Args:
        w         : [out_features, in_features] FP16/FP32 weight
        w_bit     : quantization bits (보통 4)
        group_size: 그룹 quantization 단위 (in_features 방향으로 분할)
        zero_point: True → asymmetric (zero-point 사용), False → symmetric
        clip_search: True면 그룹별로 min/max를 얼마나 잘라낼지(clip ratio) 탐색.
                     outlier 하나가 그룹 전체 scale을 키워 해상도를 낭비하는 것을 방지
        clip_ratios: 탐색할 clip 비율 후보

    Returns:
        w_dequant : quantize → dequantize된 FP weight (shape 동일)
        scale     : [out_features, n_groups] quantization scale
        zero      : [out_features, n_groups] zero-point (zero_point=False이면 None)

    # TODO: 아래 구현부를 완성하세요.
    #
    # 힌트:
    #   - w를 [out_features, n_groups, group_size] 로 reshape
    #   - 각 그룹의 min/max로 scale, zero_point 계산
    #     symmetric:  scale = max(|w|) / (2^(w_bit-1) - 1)
    #     asymmetric: scale = (max - min) / (2^w_bit - 1)
    #                 zero  = round(-min / scale)
    #   - round() + clamp() 로 quantize
    #   - dequantize: w_dequant = (w_int - zero) * scale
    """
    out_features, in_features = w.shape
    assert in_features % group_size == 0, f"in_features({in_features})가 group_size({group_size})로 나누어지지 않습니다."
    n_groups = in_features // group_size

    w = w.reshape(out_features, n_groups, group_size)

    ratios = clip_ratios if clip_search else (1.0,)
    best_err = None
    best = None  # (w_dequant, scale, zero)

    for ratio in ratios:
        if zero_point:
            w_max = w.amax(dim=-1, keepdim=True) * ratio
            w_min = w.amin(dim=-1, keepdim=True) * ratio
            q_max = (1 << w_bit) - 1  # 15 for INT4
            scale = (w_max - w_min) / q_max
            scale = scale.clamp(min=1e-8)
            zero = (-w_min / scale).round().clamp(0, q_max)
            w_int = (w / scale + zero).round().clamp(0, q_max)
            w_dq = (w_int - zero) * scale
        else:
            w_abs_max = w.abs().amax(dim=-1, keepdim=True) * ratio
            q_max = (1 << (w_bit - 1)) - 1  # 7 for INT4
            scale = w_abs_max / q_max
            scale = scale.clamp(min=1e-8)
            zero = None
            w_int = (w / scale).round().clamp(-q_max, q_max)
            w_dq = w_int * scale

        if not clip_search:
            best = (w_dq, scale, zero)
            break

        # 그룹별 재구성 오차로 최적 ratio 선택
        err = (w_dq - w).pow(2).sum(dim=-1, keepdim=True)  # [out, n_groups, 1]
        if best_err is None:
            best_err = err
            best = (w_dq, scale, zero)
        else:
            better = err < best_err  # [out, n_groups, 1]
            best_err = torch.where(better, err, best_err)
            b_dq, b_scale, b_zero = best
            b_dq = torch.where(better, w_dq, b_dq)
            b_scale = torch.where(better, scale, b_scale)
            if zero_point:
                b_zero = torch.where(better, zero, b_zero)
            best = (b_dq, b_scale, b_zero)

    w_dequant, scale, zero = best
    w_dequant = w_dequant.reshape(out_features, in_features)
    scale = scale.squeeze(-1)  # [out_features, n_groups]
    if zero is not None:
        zero = zero.squeeze(-1)

    return w_dequant, scale, zero


# ---------------------------------------------------------------------------
# AWQ Scale 탐색
# ---------------------------------------------------------------------------

def search_best_scale(
    w: torch.Tensor,
    act_scales: torch.Tensor,
    w_bit: int = 4,
    group_size: int = 128,
    zero_point: bool = True,
    n_grid: int = 20,
) -> torch.Tensor:
    """
    AWQ의 핵심: 최적 per-channel scale factor s를 grid search로 찾습니다.

    개념:
      - 원래 weight W에 대해 스케일링된 weight W_s = W * diag(s) 를 quantize
      - quantization 오류 ||W_s_dequant - W_s||를 최소화하는 s를 찾음
      - 단, s는 activation magnitude (act_scales) 기반으로 탐색 범위를 정함

    수식:
      s* = argmin_s  ||quant(W * diag(s)) - W * diag(s)||_F
      탐색 범위: s ∈ [act_scales^0, act_scales^1], grid step = 1/n_grid

    Args:
        w          : [out_features, in_features] FP weight
        act_scales : [in_features] 채널별 activation abs mean (calibration에서 수집)
        w_bit      : quantization bits
        group_size : 그룹 사이즈
        zero_point : asymmetric 여부
        n_grid     : scale 탐색 grid 수

    Returns:
        best_scale : [in_features] 최적 scale factor

    # TODO: 아래 구현부를 완성하세요.
    #
    # 힌트:
    #   - alpha를 0 ~ 1 사이로 grid search: s = act_scales^alpha
    #   - 각 alpha에 대해 W_scaled = W * s, quantize 후 오류 계산
    #   - 오류가 최소인 alpha(→ scale)를 반환
    """
    act_scales = act_scales.to(device=w.device, dtype=w.dtype)
    act_scales = act_scales.clamp(min=1e-8)

    best_error = float("inf")
    best_scale = torch.ones_like(act_scales)

    for i in range(n_grid + 1):
        alpha = i / n_grid
        scale_candidate = act_scales.pow(alpha)

        w_scaled = w * scale_candidate.unsqueeze(0)  # [out, in] * [1, in]
        w_dequant, _, _ = pseudo_quantize_tensor(
            w_scaled, w_bit=w_bit, group_size=group_size, zero_point=zero_point,
        )
        error = (w_dequant - w_scaled).abs().mean().item()

        if error < best_error:
            best_error = error
            best_scale = scale_candidate

    return best_scale


# ---------------------------------------------------------------------------
# AWQ Linear 레이어 변환
# ---------------------------------------------------------------------------

def awq_quantize_linear(
    linear: nn.Linear,
    act_scales: torch.Tensor,
    w_bit: int = 4,
    group_size: int = 128,
    zero_point: bool = True,
    clip_search: bool = False,
) -> dict:
    """
    단일 nn.Linear 레이어에 AWQ quantization을 적용합니다.

    반환값에는 quantized weight, scale, zero-point, 그리고
    vLLM 로드에 필요한 AWQ 포맷 메타데이터가 포함됩니다.

    Args:
        linear     : 원본 nn.Linear
        act_scales : [in_features] calibration에서 얻은 activation scale
        w_bit      : bits
        group_size : group size
        zero_point : asymmetric 여부

    Returns:
        {
          "qweight": INT4 packed weight tensor,
          "scales" : dequant scale,
          "zeros"  : zero-point (또는 None),
          "best_scale": AWQ scale factor (s*)
        }

    # TODO: 아래 구현부를 완성하세요.
    #
    # 단계:
    #   1. search_best_scale() 로 최적 s 탐색
    #   2. W_scaled = W * diag(s) 적용
    #   3. pseudo_quantize_tensor() 로 quantize → scale, zero 획득
    #   4. pack_int4_weight() (pack.py) 로 INT4 packing
    #   5. 반환 딕셔너리 구성
    """
    from .pack import pack_int4_weight

    w = _materialize_weight(linear).cpu()

    best_scale = search_best_scale(
        w, act_scales.cpu(), w_bit=w_bit, group_size=group_size, zero_point=zero_point,
    )

    # AWQ: scale → quantize → unscale로 최적 FP16 weight 생성
    w_scaled = w * best_scale.unsqueeze(0)
    w_dequant_scaled, _, _ = pseudo_quantize_tensor(
        w_scaled, w_bit=w_bit, group_size=group_size, zero_point=zero_point,
        clip_search=clip_search,
    )
    w_final = w_dequant_scaled / best_scale.unsqueeze(0)
    del w, w_scaled, w_dequant_scaled

    # w_final을 INT4로 양자화하여 export용 패킹
    w_dequant_final, scale, zero = pseudo_quantize_tensor(
        w_final, w_bit=w_bit, group_size=group_size, zero_point=zero_point,
        clip_search=clip_search,
    )

    out_features, in_features = w_final.shape
    n_groups = in_features // group_size
    w_reshaped = w_final.reshape(out_features, n_groups, group_size)
    del w_final
    scale_expanded = scale.unsqueeze(-1)

    if zero_point and zero is not None:
        zero_expanded = zero.unsqueeze(-1)
        w_int = (w_reshaped / scale_expanded + zero_expanded).round().clamp(0, (1 << w_bit) - 1)
    else:
        q_max = (1 << (w_bit - 1)) - 1
        w_int = (w_reshaped / scale_expanded).round().clamp(-q_max, q_max)
    del w_reshaped, scale_expanded

    w_int = w_int.reshape(out_features, in_features).to(torch.int32)
    w_int_T = w_int.T.contiguous()
    del w_int
    qweight = pack_int4_weight(w_int_T, w_bit=w_bit)
    del w_int_T

    return {
        "qweight": qweight,
        "scales": scale,
        "zeros": zero,
        "best_scale": best_scale,
        "w_dequant": w_dequant_final,
    }


# ---------------------------------------------------------------------------
# 모델 전체 AWQ 적용
# ---------------------------------------------------------------------------

def quantize_model(
    model: nn.Module,
    act_stats: dict[str, torch.Tensor],
    config: dict,
) -> nn.Module:
    """
    모델의 모든 Linear 레이어에 AWQ를 순차적으로 적용합니다.

    Args:
        model     : 원본 FP16 모델
        act_stats : calibration.py에서 얻은 {layer_name: act_scale} 딕셔너리
        config    : config.yaml 설정

    Returns:
        quantized_model: AWQ가 적용된 모델
                         (실제 INT4 커널은 export.py에서 vLLM 포맷으로 변환)

    # TODO: 레이어 순회 및 awq_quantize_linear 호출 로직을 완성하세요.
    #
    # 힌트:
    #   - model.named_modules()로 순회
    #   - layer_name이 act_stats에 있는 Linear만 처리
    #   - awq_quantize_linear() 결과를 모델에 반영
    #     (AWQLinear 같은 커스텀 모듈로 교체하거나, weight를 직접 치환)
    """
    awq_cfg = config["awq"]
    w_bit = awq_cfg["w_bit"]
    group_size = awq_cfg["group_size"]
    zero_point = awq_cfg["zero_point"]
    clip_search = awq_cfg.get("clip_search", False)

    quant_results = {}

    skip_layers = awq_cfg.get("skip_layers", {"lm_head"})
    for name, module in tqdm(list(model.named_modules()), desc="AWQ Quantizing"):
        if not isinstance(module, nn.Linear):
            continue
        if name not in act_stats:
            continue
        if name in skip_layers:
            continue
        if module.in_features % group_size != 0:
            print(f"  [skip] {name}: in_features({module.in_features})가 "
                  f"group_size({group_size})로 나누어지지 않아 FP16으로 유지합니다.")
            continue

        result = awq_quantize_linear(
            module, act_stats[name],
            w_bit=w_bit, group_size=group_size, zero_point=zero_point,
            clip_search=clip_search,
        )

        # weight를 dequantized 값으로 치환 (추론 호환성 유지)
        device = module.weight.device if module.weight.device.type != "meta" else "cpu"
        module.weight.data = result.pop("w_dequant").to(device)
        quant_results[name] = result

    return model, quant_results


# ---------------------------------------------------------------------------
# 메인 (단독 실행 테스트용)
# ---------------------------------------------------------------------------

def main():
    """작은 더미 레이어로 quantize 파이프라인을 테스트합니다."""
    torch.manual_seed(42)
    config = load_config()

    # 더미 Linear
    linear = nn.Linear(256, 512, bias=False)
    linear.weight.data = torch.randn(512, 256) * 0.02

    # 더미 activation scale
    act_scales = torch.rand(256) * 0.5 + 0.01

    print("Testing pseudo_quantize_tensor...")
    w = linear.weight.data.clone()
    w_dq, scale, zero = pseudo_quantize_tensor(w, w_bit=4, group_size=128)
    print(f"  quant error: {(w_dq - w).abs().mean():.6f}")

    print("Testing search_best_scale...")
    best_scale = search_best_scale(w, act_scales)
    print(f"  best_scale: min={best_scale.min():.4f}, max={best_scale.max():.4f}")

    print("Done.")


if __name__ == "__main__":
    main()
