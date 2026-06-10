# Jetson Orin Nano 이식 (Phase 2)

데스크톱에서 양자화한 결과물을 Jetson Orin Nano(8GB)에 올려 측정하는 단계.
**아직 실하드웨어 미검증** — 아래는 설계와 주의점.

## 무엇을 올리나

| 모델 | Jetson 8GB 적재 | 비고 |
|------|:---:|------|
| 11B fp16 | ❌ (~22GB) | 못 올림 → 정확도는 데스크톱에서 측정 |
| **11B GPTQ-4bit** | △ 검증 필요 | 텍스트 레이어만 4bit, 비전 타워는 fp16 → VRAM 빡빡할 수 있음 |
| **SmolVLM2-2.2B (무양자화)** | ✅ ~4.5GB | 엣지 참고선 |

> ⚠️ 11B는 mllama 특성상 비전 인코더/크로스어텐션이 fp16으로 남는다.
> 4bit여도 8GB에 딱 맞는지는 **실제 적재로 확인 필요**(Phase 2 핵심 리스크).

## 측정

```bash
# 무양자화 소형 모델 (가장 먼저 — 확실히 돌아감)
python scripts/jetson/bench_jetson.py --kind smolvlm --path HuggingFaceTB/SmolVLM2-2.2B-Instruct

# 양자화한 11B (적재 가능성 검증)
python scripts/jetson/bench_jetson.py --kind gptq --path models/Llama-3.2-11B-Vision-Instruct-gptq-4bit
```

측정값(peak VRAM / latency / tokens·s)은 `results/jetson_*.json` 에 저장된다.
데스크톱 `results/eval_comparison.json` 과 합쳐 최종 비교표를 만든다.

## 런타임 주의 (aarch64)

- `torch`/`torchvision` 은 **Jetson(JetPack)용 빌드**를 써야 함 (NVIDIA 제공 휠).
- `gptqmodel` 의 GPU 커널이 Jetson에서 안 맞을 수 있음. 그 경우 대안:
  - **llama.cpp + GGUF**: SmolVLM2·일부 VLM은 GGUF 변환 후 `llama.cpp`로 구동 가능.
    이쪽이 엣지 구동 안정성은 더 높음.
- 메모리 부족 시: `max_new_tokens` 축소, 이미지 해상도 축소, swap 확보.
