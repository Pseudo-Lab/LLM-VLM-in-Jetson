"""프루닝 보정(calibration) 데이터 준비."""
from __future__ import annotations

import torch


def tokenize_texts(texts: list[str], tokenizer, seq_len: int, return_mask: bool = False):
    """텍스트 리스트를 고정 길이 배치로 토크나이즈.

    return_mask=False(기본): input_ids Tensor만 반환(하위호환).
    return_mask=True: (input_ids, attention_mask) 반환 — BI 측정 시 패딩 제외용.
    """
    if tokenizer.pad_token is None:
        # Llama 계열 토크나이저는 pad_token이 없는 경우가 많음 → eos로 대체
        tokenizer.pad_token = tokenizer.eos_token
    enc = tokenizer(
        texts,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=seq_len,
    )
    if return_mask:
        return enc["input_ids"], enc["attention_mask"]
    return enc["input_ids"]


def load_korean_texts(dataset_names: list[str], n_samples: int, seed: int = 42) -> list[str]:
    """한국어 보정 코퍼스에서 텍스트 샘플을 모은다 (GPU 서버/네트워크 필요).

    KoCommercial / KoAlpaca-RealQA / kowikitext-qa의 instruction·output 필드를
    하나의 텍스트로 합쳐 반환한다.
    """
    from datasets import load_dataset

    texts: list[str] = []
    per = max(1, n_samples // len(dataset_names))
    for name in dataset_names:
        ds = load_dataset(name, split="train", streaming=True)
        for i, row in enumerate(ds):
            if i >= per:
                break
            texts.append(_row_to_text(row))
    return texts[:n_samples]


def _row_to_text(row: dict) -> str:
    """데이터셋 행에서 학습용 텍스트를 추출 (필드명 차이 흡수)."""
    for key in ("text", "instruction", "question", "input"):
        if key in row and row[key]:
            extra = row.get("output") or row.get("answer") or row.get("response") or ""
            return f"{row[key]}\n{extra}".strip()
    return str(next(iter(row.values())))
