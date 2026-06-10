"""GPTQ 4비트 양자화 본체 — Llama 3.2 11B Vision-Instruct (mllama).

중요 사실(검증됨):
  GPTQModel 의 mllama 지원은 "텍스트(언어) 레이어 한정"이다. 비전 인코더와
  크로스어텐션은 fp16 으로 남고 LLM 디코더 레이어만 4bit 로 양자화된다.
  → 캘리브레이션은 텍스트(list[str])만으로 충분하며 공식 예제와 일치한다.
  → Jetson 적재 시 비전 타워는 fp16 그대로이므로 VRAM 계산에 반영할 것.

API (GPTQModel 최신):
  from gptqmodel import GPTQModel, QuantizeConfig
  model = GPTQModel.load(model_id, quant_config)
  model.quantize(calibration_dataset, batch_size=1)
  model.save(quant_path)

실행:
  python src/quantize.py
세부 설정은 configs/gptq_config.yaml 에서 조정.
"""
import mllama_compat  # noqa: F401  gptqmodel×transformers5 보정 — gptqmodel 보다 먼저!

from gptqmodel import GPTQModel, QuantizeConfig

from config import load_config, resolve
from calibration import build_text_calibration, build_calibration


def _build_quant_config(qcfg: dict) -> QuantizeConfig:
    return QuantizeConfig(
        bits=qcfg["bits"],
        group_size=qcfg["group_size"],
        desc_act=qcfg.get("desc_act", False),
        sym=qcfg.get("sym", True),
        # meta-shell(turtle) 로더는 transformers 5.x mllama 의 마스크 생성
        # (attention_mask.all() — 실값 필요)과 충돌해 죽는다. RAM 64GB 라
        # 직접 CPU 로드로 충분하므로 디스크 오프로드를 끈다.
        offload_to_disk=qcfg.get("offload_to_disk", False),
    )


def _resolve_source(mcfg: dict) -> str:
    """로컬 다운로드본이 있으면 그 경로를, 없으면 HF repo id 를 사용."""
    local = resolve(mcfg["local_dir"])
    if local.exists() and any(local.iterdir()):
        print(f"[quantize] 로컬 모델 사용: {local}")
        return str(local)
    print(f"[quantize] 로컬 다운로드본 없음 → HF 에서 직접 로드: {mcfg['id']}")
    return mcfg["id"]


def main() -> None:
    cfg = load_config()
    mcfg, qcfg, ccfg = cfg["model"], cfg["quantize"], cfg["calibration"]

    source = _resolve_source(mcfg)
    out_dir = resolve(qcfg["output_dir"])
    quant_config = _build_quant_config(qcfg)

    print(
        f"[quantize] bits={qcfg['bits']} group_size={qcfg['group_size']} "
        f"desc_act={quant_config.desc_act} sym={quant_config.sym}"
    )

    # 1) 모델 로드 (12GB VRAM: GPTQModel 이 레이어 단위로 처리, 나머지는 CPU 오프로드)
    model = GPTQModel.load(source, quant_config)

    # 2) 캘리브레이션 데이터 준비
    mode = ccfg.get("mode", "text")
    if mode == "multimodal":
        print("[quantize] 멀티모달 캘리브(실험적) 사용")
        calibration = build_calibration(model.preprocessor or model.tokenizer)
    else:
        calibration = build_text_calibration()

    # 3) 양자화
    print(f"[quantize] 양자화 시작 (samples={len(calibration)}) — 시간 소요됩니다…")
    model.quantize(calibration, batch_size=cfg["runtime"].get("batch_size", 1))

    # 4) 저장 (가중치 + processor/tokenizer 함께)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save(str(out_dir))
    print(f"[quantize] 완료 → {out_dir}")


if __name__ == "__main__":
    main()
