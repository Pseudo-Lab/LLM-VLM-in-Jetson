import torch
from transformers import AutoTokenizer
from src.prune.calibration import tokenize_texts


def test_tokenize_texts_shape():
    tok = AutoTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer")
    texts = ["안녕하세요 반갑습니다", "오늘 날씨가 좋네요", "경량화 테스트 문장"]
    batch = tokenize_texts(texts, tok, seq_len=8)
    assert isinstance(batch, torch.Tensor)
    assert batch.shape[1] == 8
    assert batch.shape[0] == 3
