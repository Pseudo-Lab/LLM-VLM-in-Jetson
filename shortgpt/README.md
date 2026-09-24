# Phi-4 경량화 (Jetson Orin Nano 8GB)

Minitron 구조적 프루닝 → distillation → GGUF 양자화 파이프라인.

## 셋업
    cd shortgpt && python3.10 -m venv .venv
    .venv/bin/pip install -r requirements.txt

## 테스트
    .venv/bin/python -m pytest            # 단위(로컬, tiny 모델)
    .venv/bin/python -m pytest -m integration   # 실모델(GPU 서버)

## 프루닝 실행 (GPU 서버)
    .venv/bin/python scripts/run_prune.py --config configs/prune_phi4.yaml
