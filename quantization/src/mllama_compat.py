"""gptqmodel 7.1.0 × transformers 5.x mllama 호환 보정 — import 만으로 적용.

gptqmodel 을 쓰는 모든 진입점(quantize/benchmark/테스트)은 gptqmodel 사용 전에
이 모듈을 import 할 것. 두 가지를 보정한다:

1. pyarrow DLL 충돌 (Windows): gptqmodel import 도중 내부에서 datasets→pyarrow 가
   로드되면 access violation 으로 segfault. datasets 를 먼저 로드해 회피.
2. module_tree 경로: gptqmodel 7.1.0 의 mllama 정의는 transformers 4.x 구조
   (language_model.model.layers) 기준이라 5.x 에서 layers=None 으로 양자화·로드가
   모두 죽는다. 5.x 실제 구조(model.language_model.layers)로 교체
   (동일 버전 qwen3_vl 정의와 같은 경로 형태). 양자화와 from_quantized 로드
   양쪽이 이 경로를 쓴다.
"""
import os

# (0) Windows: triton 이 없어 torch.compile(inductor) GPU 컴파일이 TritonMissing 으로
#     죽는다 (gptqmodel torch 커널의 dequantize_weight 가 @torch.compile). dynamo 를
#     끄면 eager 로 폴백 — 느리지만 동작.
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import datasets  # noqa: F401  (1) — import 순서 자체가 목적

from gptqmodel.models.definitions.mllama import MLlamaQModel

# (2) — module_tree 의 마지막 원소(레이어 내부 모듈 dict)는 그대로 재사용
MLlamaQModel.module_tree = [
    "model", "language_model", "layers", "#", MLlamaQModel.module_tree[-1]
]
MLlamaQModel.pre_lm_head_norm_module = "model.language_model.norm"
