from dataclasses import dataclass, field
from typing import Literal, Optional


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to prune/retrain.
    Shared by search_pruning_layer.py and replace_and_retrain.py.
    """

    model_name: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"},
        default="meta-llama/Llama-3.1-8B"
    )

    use_fast: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )


@dataclass
class SearchArguments:
    """
    Arguments for search_pruning_layer.py — locating the [pruning_start_layer,
    pruning_end_layer] range to prune via cosine similarity.
    """

    layer_intervals: Optional[int] = field(
        default=8,
        metadata={"help": "Number of consecutive layers to prune."},
    )

    cosine_num_data: Optional[int] = field(
        default=50,
        metadata={"help": "Amount of data used to calculate cosine similarity."},
    )

    train_num_data: Optional[int] = field(
        default=100000,
        metadata={"help": "Amount of data used to build the dataset the cosine similarity search samples from."},
    )


@dataclass
class TrainingArguments:
    """
    Arguments for replace_and_retrain.py — retraining a single replacement layer
    for a pruned block of layers.
    """

    pruning_start_layer: int = field(
        metadata={"help": "0-indexed first layer to prune (inclusive)."},
    )

    pruning_end_layer: int = field(
        metadata={"help": "0-indexed last layer to prune (inclusive). "
                           "replace_layer approximates layers [pruning_start_layer ... pruning_end_layer]."},
    )

    model: Literal["llama", "opt"] = field(
        default="llama",
        metadata={"help": "Model family to retrain."},
    )

    replace: Literal["none", "mlp", "tf"] = field(
        default="tf",
        metadata={"help": "Replacement layer type."},
    )

    data_path: Optional[str] = field(
        default=None,
        metadata={"help": "Local path for load_from_disk. If None, downloads SlimPajama-6B."},
    )

    output_dir: Optional[str] = field(
        default="./pruned_model",
    )

    batch_size: Optional[int] = field(default=8)

    grad_accum: Optional[int] = field(default=16)

    epochs: Optional[int] = field(default=1)

    eval_every: Optional[int] = field(
        default=500,
        metadata={"help": "Evaluate every N gradient steps."},
    )

    patience: Optional[int] = field(
        default=5,
        metadata={"help": "Stop retraining after this many consecutive evals with no new best eval loss. Set 0 to disable."},
    )

    lr: float = field(default=2e-4)
    min_lr: float = field(default=5e-5)
    weight_decay: float = field(default=1e-3)


@dataclass
class BenchmarkArguments:
    """
    Arguments for benchmark.py — 1-shot log-likelihood evaluation on KMMLU
    (https://huggingface.co/datasets/HAERAE-HUB/KMMLU), reused by all model/replace
    combinations. `ModelArguments.model_name` doubles as the checkpoint path here,
    same as in search_pruning_layer.py / replace_and_retrain.py.

    num_fewshot defaults to 1, not the paper's 5: OPT's English-trained BPE
    tokenizes Korean far more densely than English, so measured against
    facebook/opt-6.7b's own tokenizer, a 5-shot prompt overflows its 2048-token
    max_position_embeddings on ~76% of KMMLU questions (silently discarding
    most/all of the fewshot exemplars for those), vs. ~6% at 1-shot.
    """

    output_dir: str = field(
        default="./benchmark_results",
        metadata={"help": "Directory for kmmlu_summary.csv and <model_label>_per_subject.csv."},
    )

    model_label: Optional[str] = field(
        default=None,
        metadata={"help": "Value written to the CSV 'Model' column. Defaults to model_name's basename."},
    )

    replace_type: Optional[str] = field(
        default=None,
        metadata={"help": "Value written to the CSV 'Replacement' column (e.g. none/mlp/tf); informational only."},
    )

    num_fewshot: int = field(
        default=1,
        metadata={"help": "Few-shot exemplars drawn from each subject's KMMLU 'dev' split (up to 5 available per subject)."},
    )

    max_examples: Optional[int] = field(
        default=None,
        metadata={"help": "Cap test examples scored per subject. Peak VRAM is set by one score_choices "
                           "batch (4 sequences, bounded by max_position_embeddings), not by dataset size, "
                           "so a small cap (e.g. 5-10) still yields a representative Peak VRAM reading "
                           "while running in a fraction of the time of a full accuracy pass."},
    )

    dtype: Literal["bf16", "fp16", "fp32"] = field(
        default="bf16",
        metadata={"help": "torch_dtype used to load the model for inference."},
    )
