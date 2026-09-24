"""configs/gptq_config.yaml 로딩 + 공통 경로 헬퍼."""
from pathlib import Path
import os

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")


def load_config(path: str | Path = ROOT / "configs" / "gptq_config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def hf_token() -> str | None:
    """gated 모델 접근용 HF 토큰."""
    return os.environ.get("HF_TOKEN")


def resolve(path: str) -> Path:
    """상대 경로를 프로젝트 루트 기준 절대 경로로."""
    p = Path(path)
    return p if p.is_absolute() else ROOT / p
