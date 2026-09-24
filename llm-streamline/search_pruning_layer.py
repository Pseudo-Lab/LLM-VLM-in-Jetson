"""
Search the pruning layer range via cosine similarity only — no lightweight-network
retraining involved. The block [pruning_start_layer, pruning_end_layer] (inclusive)
is dropped outright, so it spans training_args.layer_intervals layers exactly (no
extra layer is inserted to compensate).

Usage:
    python search_pruning_layer.py \
    --model_name facebook/opt-6.7b \
    --layer_intervals 8

Output:
    pruning_start_layer / pruning_end_layer, the 0-indexed inclusive [start, end]
    range of layers to delete.
"""

from transformers import (
    AutoModelForCausalLM,
    AutoConfig,
    AutoTokenizer,
    HfArgumentParser,
)
from datasets import load_dataset

from args import ModelArguments, SearchArguments
from LLM_Streamline.get_cosine import get_cosine_similarity
from LLM_Streamline.train_lightweightnetwork import process_datasets


def parse_hf_args():
    parser = HfArgumentParser((ModelArguments, SearchArguments))
    args, search_args, _ = parser.parse_args_into_dataclasses(
        return_remaining_strings=True)

    return args, search_args


def run():
    args, search_args = parse_hf_args()

    model = AutoModelForCausalLM.from_pretrained(args.model_name, trust_remote_code=True)
    config = AutoConfig.from_pretrained(args.model_name, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset('DKYoon/SlimPajama-6B')['train']
    dataset, _ = process_datasets(dataset, search_args.train_num_data, tokenizer)

    best_layer = get_cosine_similarity(
        model, dataset, search_args.cosine_num_data, 'cuda',
        search_args.layer_intervals, config.num_hidden_layers)

    pruning_start_layer = best_layer
    pruning_end_layer = best_layer + search_args.layer_intervals - 1

    print(f"\n\npruning_layer: [{pruning_start_layer}, {pruning_end_layer}]")

    return pruning_start_layer, pruning_end_layer

if __name__ == "__main__":
    run()
