"""원본 Llama 3.2 11B Vision 모델 다운로드.

gated 모델이므로 .env 의 HF_TOKEN 과 라이선스 승인이 선행되어야 함.
"""
from huggingface_hub import snapshot_download

from config import load_config, hf_token, resolve


def main() -> None:
    cfg = load_config()
    model_id = cfg["model"]["id"]
    local_dir = resolve(cfg["model"]["local_dir"])

    token = hf_token()
    if not token:
        raise SystemExit(
            "HF_TOKEN 이 없습니다. .env.example 을 .env 로 복사하고 토큰을 넣으세요.\n"
            "그리고 https://huggingface.co/meta-llama/Llama-3.2-11B-Vision-Instruct "
            "에서 라이선스 승인이 필요합니다."
        )

    local_dir.mkdir(parents=True, exist_ok=True)
    print(f"[download] {model_id} -> {local_dir}")
    snapshot_download(
        repo_id=model_id,
        local_dir=str(local_dir),
        token=token,
        ignore_patterns=["*.pth", "original/*"],  # consolidated 원본 중복 제외
    )
    print("[download] 완료")


if __name__ == "__main__":
    main()
