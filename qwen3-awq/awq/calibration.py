"""
awq/calibration.py

AWQ에서 사용하는 calibration 데이터를 수집하고,
각 Linear 레이어의 입력 activation 통계를 기록합니다.

AWQ 핵심 아이디어:
  - weight quantization 오류는 모든 채널에 동등하지 않음
  - activation magnitude가 큰 채널(salient channel)의 오류가 최종 출력에 더 큰 영향을 미침
  - 따라서 activation 통계를 먼저 수집해야 함
"""

import torch
import yaml
from typing import Optional
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM


def load_config(config_path: str = "../configs/config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Calibration 데이터셋 로드
# ---------------------------------------------------------------------------

def get_calib_dataset(
    dataset_name: str = "pileval",
    tokenizer=None,
    n_samples: int = 512,
    seq_len: int = 512,
) -> list[torch.Tensor]:
    """
    Calibration에 사용할 토큰 시퀀스를 반환합니다.

    Args:
        dataset_name: 사용할 데이터셋 ("pileval", "wikitext2", "c4")
        tokenizer: HuggingFace tokenizer
        n_samples: 수집할 샘플 수
        seq_len: 각 샘플의 시퀀스 길이

    Returns:
        [n_samples, seq_len] 크기의 input_ids 텐서 리스트
    """
    if dataset_name == "pileval":
        dataset = load_dataset("mit-han-lab/pile-val-backup", split="validation")
        texts = [item["text"] for item in dataset]
    elif dataset_name == "wikitext2":
        dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
        texts = [item["text"] for item in dataset if len(item["text"]) > 100]
    elif dataset_name == "kowikitext":
        # 한국어 위키피디아 (Parquet 포맷 — 스크립트 기반이 아니라 최신 datasets에서 동작)
        # streaming으로 필요한 만큼만 받음
        dataset = load_dataset("wikimedia/wikipedia", "20231101.ko", split="train", streaming=True)
        texts = []
        for item in dataset:
            if len(item["text"]) > 100:
                texts.append(item["text"])
            if len(texts) >= n_samples * 4:
                break
    elif dataset_name == "c4":
        dataset = load_dataset("c4", "en", split="train", streaming=True)
        texts = [item["text"] for item in dataset.take(n_samples * 2)]
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    samples = []
    for text in texts:
        enc = tokenizer(text, return_tensors="pt", truncation=False)
        input_ids = enc.input_ids[0]

        # seq_len 단위로 자르기
        for start in range(0, len(input_ids) - seq_len, seq_len):
            samples.append(input_ids[start : start + seq_len])
            if len(samples) >= n_samples:
                return samples

    return samples


# ---------------------------------------------------------------------------
# Activation Hook으로 통계 수집
# ---------------------------------------------------------------------------

class ActivationCollector:
    """
    nn.Linear 레이어의 입력 activation을 hook으로 수집하고
    채널별 통계(mean absolute value)를 기록합니다.

    collect_inputs=True이면 출력-오차 기반 scale/clip 탐색(공식 AWQ 방식)에 쓸
    입력 서브샘플도 함께 캐시합니다. 레이어당 최대 n_input_rows 행을
    fp16/CPU로 보관하므로 메모리 부담이 작습니다
    (예: 253 레이어 × 512행 × hidden 2560 × 2B ≈ 0.7GB).
    """

    def __init__(
        self,
        collect_inputs: bool = False,
        n_input_rows: int = 512,
        n_calib_samples: int = 128,
    ):
        self.stats: dict[str, torch.Tensor] = {}   # layer_name -> abs_mean per input channel
        self._hooks = []
        self._layer_names: dict = {}               # module -> name

        self.collect_inputs = collect_inputs
        self.n_input_rows = n_input_rows
        # calibration 전체에 걸쳐 고르게 뽑히도록 forward 1회당 캐시할 행 수
        self._rows_per_call = max(1, n_input_rows // max(1, n_calib_samples))
        self._input_rows: dict[str, list[torch.Tensor]] = {}
        self._input_counts: dict[str, int] = {}

    def register(self, model: torch.nn.Module) -> None:
        """모든 Linear 레이어에 forward hook 등록."""
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                self._layer_names[module] = name
                hook = module.register_forward_hook(self._hook_fn)
                self._hooks.append(hook)

    def _hook_fn(self, module, input, output):
        """
        Forward hook: 입력 activation의 채널별 |mean| 을 누적합니다.

        input[0] shape: [batch, seq_len, in_features]  (or [batch, in_features])
        """
        x = input[0].detach().float()               # [B, T, C] or [B, C]
        if x.dim() == 3:
            x = x.view(-1, x.shape[-1])             # [B*T, C]

        abs_mean = x.abs().mean(dim=0)              # [C] — 채널별 평균 활성화 크기

        name = self._layer_names[module]
        if name not in self.stats:
            self.stats[name] = abs_mean
        else:
            # 누적 평균 (온라인 방식)
            self.stats[name] = (self.stats[name] + abs_mean) / 2.0

        if self.collect_inputs and self._input_counts.get(name, 0) < self.n_input_rows:
            k = min(self._rows_per_call, x.shape[0],
                    self.n_input_rows - self._input_counts.get(name, 0))
            idx = torch.randperm(x.shape[0], device=x.device)[:k]
            rows = x[idx].to(torch.float16).cpu()
            self._input_rows.setdefault(name, []).append(rows)
            self._input_counts[name] = self._input_counts.get(name, 0) + k

    def get_input_cache(self) -> dict[str, torch.Tensor]:
        """레이어별로 캐시한 입력 행들을 [n_rows, in_features] 텐서로 합쳐 반환."""
        return {name: torch.cat(rows, dim=0) for name, rows in self._input_rows.items()}

    def remove(self) -> None:
        """등록된 hook 제거."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()


# ---------------------------------------------------------------------------
# Calibration 실행
# ---------------------------------------------------------------------------

def run_calibration(
    model: torch.nn.Module,
    tokenizer,
    config: dict,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """
    Calibration 데이터를 forward pass하여 각 레이어의
    입력 activation 통계를 수집합니다.

    config["calibration"]["collect_inputs"]가 True이면 출력-오차 탐색용
    입력 서브샘플 캐시도 함께 수집합니다.

    Args:
        model: 원본 FP16 모델
        tokenizer: HuggingFace tokenizer
        config: 설정 딕셔너리

    Returns:
        (act_stats, input_cache)
          act_stats  : layer_name -> abs_mean_per_channel
          input_cache: layer_name -> [n_rows, in_features] 입력 서브샘플
                       (collect_inputs=False이면 빈 딕셔너리)
    """
    calib_cfg = config["calibration"]
    device = next(model.parameters()).device

    print(f"Loading calibration dataset: {calib_cfg['dataset']}")
    samples = get_calib_dataset(
        dataset_name=calib_cfg["dataset"],
        tokenizer=tokenizer,
        n_samples=calib_cfg["n_samples"],
        seq_len=calib_cfg["seq_len"],
    )

    collector = ActivationCollector(
        collect_inputs=calib_cfg.get("collect_inputs", False),
        n_input_rows=calib_cfg.get("n_input_rows", 512),
        n_calib_samples=len(samples),
    )
    collector.register(model)
    model.eval()

    print(f"Running {len(samples)} calibration samples...")
    with torch.no_grad():
        for input_ids in tqdm(samples, desc="Calibration"):
            input_ids = input_ids.unsqueeze(0).to(device)   # [1, seq_len]
            model(input_ids)

    collector.remove()

    print(f"Collected activation stats for {len(collector.stats)} layers.")
    input_cache = collector.get_input_cache() if collector.collect_inputs else {}
    if input_cache:
        total_mb = sum(t.numel() * t.element_size() for t in input_cache.values()) / 1e6
        print(f"Cached input subsamples for {len(input_cache)} layers ({total_mb:.0f} MB).")
    return collector.stats, input_cache


# ---------------------------------------------------------------------------
# 메인
# ---------------------------------------------------------------------------

def main():
    config = load_config()
    model_name = config["paths"]["baseline_model"]

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    act_stats, _ = run_calibration(model, tokenizer, config)

    # 확인용 출력
    for name, stat in list(act_stats.items())[:5]:
        print(f"{name}: shape={stat.shape}, max={stat.max():.4f}, mean={stat.mean():.4f}")


if __name__ == "__main__":
    main()
