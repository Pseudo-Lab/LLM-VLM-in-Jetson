"""
awq/pack.py

INT4 weight packing 유틸리티.

GPU 메모리 효율을 위해 INT4 두 값을 INT8 하나에 패킹합니다.
vLLM의 AWQ 커널이 기대하는 포맷(GEMM-friendly layout)으로 변환합니다.

패킹 방식 (AWQ 표준):
  - in_features 방향을 group_size 단위로 분할
  - 각 그룹 내 weight를 INT4로 양자화 후 두 값씩 묶어 INT8로 저장
  - shape 변환: [out_features, in_features] → [out_features, in_features // 8 * w_bit]
"""

import torch

# AWQ GEMM 커널의 인터리브 패킹 순서:
# INT32의 i번째 nibble(4bit)에 원본 인덱스 AWQ_PACK_ORDER[i]의 값이 들어감
AWQ_PACK_ORDER = [0, 2, 4, 6, 1, 3, 5, 7]


def pack_int4_weight(
    w_int: torch.Tensor,
    w_bit: int = 4,
) -> torch.Tensor:
    """
    INT4 양자화된 weight를 INT32 텐서에 패킹합니다.

    vLLM AWQ 커널은 INT4 값 8개를 INT32 하나에 묶는 포맷을 사용합니다.

    Args:
        w_int : [out_features, in_features] INT4 범위 (0~15)의 정수 텐서
        w_bit : quantization bits (현재 4만 지원)

    Returns:
        qweight: [out_features, in_features // (32 // w_bit)] INT32 텐서

    # TODO: 아래 구현부를 완성하세요.
    #
    # 힌트:
    #   - values_per_int32 = 32 // w_bit  → INT4면 8
    #   - in_features를 values_per_int32 로 나누어 [out_features, in_features//8, 8] reshape
    #   - bit shift로 8개 INT4를 INT32 하나로 합치기:
    #       packed = w[:,  :, 0]
    #              | (w[:, :, 1] << 4)
    #              | (w[:, :, 2] << 8)  ...
    """
    assert w_bit == 4, "현재 INT4 packing만 지원합니다."

    out_features, in_features = w_int.shape
    values_per_int32 = 32 // w_bit  # 8

    w_int = w_int.to(torch.int32)
    w_int = w_int.reshape(out_features, in_features // values_per_int32, values_per_int32)

    packed = torch.zeros(out_features, in_features // values_per_int32, dtype=torch.int32, device=w_int.device)
    for i in range(values_per_int32):
        packed |= (w_int[:, :, AWQ_PACK_ORDER[i]] << (i * w_bit))

    return packed


def unpack_int4_weight(
    qweight: torch.Tensor,
    w_bit: int = 4,
    original_in_features: int = None,
) -> torch.Tensor:
    """
    pack_int4_weight의 역연산 — 디버깅 및 검증용.

    Args:
        qweight              : [out_features, in_features // 8] INT32 packed tensor
        w_bit                : quantization bits
        original_in_features : 원본 in_features (None이면 qweight에서 추론)

    Returns:
        w_int: [out_features, in_features] INT4 범위 정수 텐서

    # TODO: 아래 구현부를 완성하세요.
    #
    # 힌트:
    #   - pack의 역순: INT32 → 8개 INT4로 분리
    #   - bit mask: 0xF (= 0b1111) 로 하위 4비트 추출
    #   - right shift 후 mask 적용
    """
    values_per_int32 = 32 // w_bit  # 8
    out_features = qweight.shape[0]
    packed_in = qweight.shape[1]

    if original_in_features is None:
        original_in_features = packed_in * values_per_int32

    mask = (1 << w_bit) - 1  # 0xF

    unpacked = [None] * values_per_int32
    for i in range(values_per_int32):
        unpacked[AWQ_PACK_ORDER[i]] = (qweight >> (i * w_bit)) & mask

    w_int = torch.stack(unpacked, dim=-1)  # [out_f, packed_in, 8]
    w_int = w_int.reshape(out_features, -1)[:, :original_in_features]

    return w_int


def verify_pack_unpack(out_f: int = 64, in_f: int = 128):
    """pack → unpack 왕복 검증 (구현 완료 후 테스트용)."""
    torch.manual_seed(0)
    w_int = torch.randint(0, 16, (out_f, in_f), dtype=torch.int32)

    qweight = pack_int4_weight(w_int)
    w_restored = unpack_int4_weight(qweight, original_in_features=in_f)
    assert torch.all(w_int == w_restored), "Pack/Unpack 불일치!"
    print(f"Pack/Unpack 검증 통과: {out_f}x{in_f} → packed {qweight.shape} → restored {w_restored.shape}")


if __name__ == "__main__":
    verify_pack_unpack()
