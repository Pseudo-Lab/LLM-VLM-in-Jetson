"""
1-shot, log-likelihood evaluation of a pruned checkpoint on KMMLU
(https://huggingface.co/datasets/HAERAE-HUB/KMMLU): 45 subjects, each scored by
comparing the model's log-probability of continuing a "정답：" prompt with "A"/
"B"/"C"/"D" and taking the argmax (same methodology as lm-evaluation-harness's
default "kmmlu" task group). Subjects are macro-averaged into the paper's four
reporting categories (STEM / Applied Science / HUMSS / Other) plus an overall
Average across all 45 subjects — see modeling/prune_utils.py-saved checkpoints,
loadable via AutoModelForCausalLM.from_pretrained(..., trust_remote_code=True).

num_fewshot defaults to 1, not the paper's 5: OPT's English-trained BPE
tokenizes Korean far more densely than English, so a 5-shot prompt overflows
facebook/opt-6.7b's 2048-token max_position_embeddings on ~76% of KMMLU
questions (silently discarding most/all fewshot exemplars for those), vs. ~6%
at 1-shot — see args.py's BenchmarkArguments for the measurement this is based
on. The remaining ~6% (mostly long scenario-style questions in subjects like
Criminal-Law/Patent/Accounting) still gets truncated via score_choices' left-
truncation below, regardless of num_fewshot.

Per-subject results are appended incrementally to
<output_dir>/<model_label>_kmmlu_per_subject.csv, so a killed/requeued slurm job
can resume without re-scoring already-finished subjects. The final row (one per
model) is appended to the shared <output_dir>/kmmlu_summary.csv.

Usage:
    python benchmark.py \
        --model_name /path/to/opt_prune_from2to9_mlp \
        --model_label "OPT-6.7b: 8 layer pruned" \
        --replace_type mlp \
        --output_dir ./benchmark_results
"""

import csv
import os

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser

from args import BenchmarkArguments, ModelArguments

KMMLU_CATEGORIES = {
    "Accounting": "HUMSS",
    "Agricultural-Sciences": "Other",
    "Aviation-Engineering-and-Maintenance": "Applied Science",
    "Biology": "STEM",
    "Chemical-Engineering": "STEM",
    "Chemistry": "STEM",
    "Civil-Engineering": "STEM",
    "Computer-Science": "STEM",
    "Construction": "Other",
    "Criminal-Law": "HUMSS",
    "Ecology": "STEM",
    "Economics": "HUMSS",
    "Education": "HUMSS",
    "Electrical-Engineering": "STEM",
    "Electronics-Engineering": "Applied Science",
    "Energy-Management": "Applied Science",
    "Environmental-Science": "Applied Science",
    "Fashion": "Other",
    "Food-Processing": "Other",
    "Gas-Technology-and-Engineering": "Applied Science",
    "Geomatics": "Applied Science",
    "Health": "Other",
    "Industrial-Engineer": "Applied Science",
    "Information-Technology": "STEM",
    "Interior-Architecture-and-Design": "Other",
    "Korean-History": "HUMSS",
    "Law": "HUMSS",
    "Machine-Design-and-Manufacturing": "Applied Science",
    "Management": "HUMSS",
    "Maritime-Engineering": "Applied Science",
    "Marketing": "Other",
    "Materials-Engineering": "STEM",
    "Math": "STEM",
    "Mechanical-Engineering": "STEM",
    "Nondestructive-Testing": "Applied Science",
    "Patent": "Other",
    "Political-Science-and-Sociology": "HUMSS",
    "Psychology": "HUMSS",
    "Public-Safety": "Other",
    "Railway-and-Automotive-Engineering": "Applied Science",
    "Real-Estate": "Other",
    "Refrigerating-Machinery": "Other",
    "Social-Welfare": "HUMSS",
    "Taxation": "HUMSS",
    "Telecommunications-and-Wireless-Technology": "Applied Science",
}
CHOICES = ["A", "B", "C", "D"]
DTYPE_MAP = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


def format_question(row):
    return (
        f"{row['question'].strip()}\n"
        f"A. {row['A']}\nB. {row['B']}\nC. {row['C']}\nD. {row['D']}\n정답："
    )


def build_fewshot_prefix(dev_rows, num_fewshot):
    examples = []
    for row in dev_rows.select(range(min(num_fewshot, len(dev_rows)))):
        examples.append(format_question(row) + CHOICES[row["answer"] - 1])
    return ("\n\n".join(examples) + "\n\n") if examples else ""


@torch.no_grad()
def score_choices(model, tokenizer, context, device, max_position_embeddings):
    """
    Returns (scores, truncated): scores is a length-4 list of log-probability
    sums, one per candidate in CHOICES, each the model's log-likelihood of that
    choice as a continuation of `context`; truncated flags whether context had
    to be shortened to fit the model's position-embedding table.

    OPT's tokenizer is an English-trained byte-level BPE with no Korean-aware
    merges, so Korean text tokenizes far more densely than English does on it —
    a 5-shot Korean prompt can overflow max_position_embeddings (2048 for
    OPT-6.7b) even though it looks short. Overflowing it feeds an out-of-range
    index into the learned position embedding and crashes with a CUDA
    device-side assert, so context is truncated from the left (dropping the
    earliest few-shot exemplars first) to leave room for the longest choice.
    """
    context_ids = tokenizer(context)["input_ids"]
    choice_ids = [tokenizer(c, add_special_tokens=False)["input_ids"] for c in CHOICES]
    reserve = max(len(ids) for ids in choice_ids)

    truncated = len(context_ids) > max_position_embeddings - reserve
    if truncated:
        context_ids = context_ids[-(max_position_embeddings - reserve):]

    sequences = [context_ids + ids for ids in choice_ids]
    cont_lens = [len(ids) for ids in choice_ids]

    max_len = max(len(s) for s in sequences)
    pad_id = tokenizer.pad_token_id
    input_ids = torch.full((len(CHOICES), max_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((len(CHOICES), max_len), dtype=torch.long)
    for i, s in enumerate(sequences):
        input_ids[i, max_len - len(s):] = torch.tensor(s, dtype=torch.long)
        attention_mask[i, max_len - len(s):] = 1

    logits = model(
        input_ids=input_ids.to(device), attention_mask=attention_mask.to(device)
    ).logits
    logprobs = torch.log_softmax(logits.float(), dim=-1)

    scores = []
    for i, (s, cont_len) in enumerate(zip(sequences, cont_lens)):
        pad_offset = max_len - len(s)
        cont_start = pad_offset + (len(s) - cont_len)
        score = sum(
            logprobs[i, pos - 1, input_ids[i, pos]].item()
            for pos in range(cont_start, pad_offset + len(s))
        )
        scores.append(score)
    return scores, truncated


def load_finished_subjects(per_subject_csv):
    finished = {}
    if os.path.exists(per_subject_csv):
        with open(per_subject_csv, newline="") as f:
            for row in csv.DictReader(f):
                finished[row["subject"]] = {
                    "accuracy": float(row["accuracy"]),
                    "num_examples": int(row["num_examples"]),
                }
    return finished


if __name__ == "__main__":
    parser = HfArgumentParser((ModelArguments, BenchmarkArguments))
    model_args, args, _ = parser.parse_args_into_dataclasses(return_remaining_strings=True)

    os.makedirs(args.output_dir, exist_ok=True)
    model_label = args.model_label or os.path.basename(model_args.model_name.rstrip("/"))
    per_subject_csv = os.path.join(args.output_dir, f"{model_label}_kmmlu_per_subject.csv")
    summary_csv = os.path.join(args.output_dir, "kmmlu_summary.csv")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading {model_args.model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name, use_fast=model_args.use_fast, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name, torch_dtype=DTYPE_MAP[args.dtype], trust_remote_code=True
    ).to(device)
    model.eval()
    max_position_embeddings = getattr(model.config, "max_position_embeddings", 2048)

    # Peak VRAM is set by weight loading plus one score_choices batch (4 sequences,
    # bounded by max_position_embeddings) — it doesn't grow with dataset size, so this
    # reading stays representative even under --max_examples.
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    num_params = sum(p.numel() for p in model.parameters())
    bytes_per_param = torch.finfo(DTYPE_MAP[args.dtype]).bits // 8
    model_size_gb = num_params * bytes_per_param / (1024 ** 3)
    print(f"Model parameters: {num_params:,} ({num_params / 1e9:.2f}B)  ~{model_size_gb:.2f} GB in {args.dtype}")

    results = load_finished_subjects(per_subject_csv)
    write_header = not os.path.exists(per_subject_csv)
    with open(per_subject_csv, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["subject", "category", "num_examples", "accuracy"])

        for subject in KMMLU_CATEGORIES:
            if subject in results:
                print(f"[{subject}] already scored — accuracy {results[subject]['accuracy']:.4f} (skipped)")
                continue

            dataset = load_dataset("HAERAE-HUB/KMMLU", subject)
            fewshot_prefix = build_fewshot_prefix(dataset["dev"], args.num_fewshot)
            test_rows = dataset["test"]
            if args.max_examples is not None:
                test_rows = test_rows.select(range(min(args.max_examples, len(test_rows))))

            correct = 0
            truncated_count = 0
            for row in tqdm(test_rows, desc=subject):
                context = fewshot_prefix + format_question(row)
                scores, truncated = score_choices(
                    model, tokenizer, context, device, max_position_embeddings
                )
                pred = max(range(len(CHOICES)), key=lambda i: scores[i])
                correct += int(pred == row["answer"] - 1)
                truncated_count += int(truncated)

            accuracy = correct / len(test_rows)
            results[subject] = {"accuracy": accuracy, "num_examples": len(test_rows)}
            writer.writerow([subject, KMMLU_CATEGORIES[subject], len(test_rows), accuracy])
            f.flush()
            print(
                f"[{subject}] accuracy: {accuracy:.4f} ({correct}/{len(test_rows)})"
                + (f"  [truncated context: {truncated_count}/{len(test_rows)}]" if truncated_count else "")
            )

    # ─────────────────────────────────────────────────────────────────────────
    # Macro-average per category + overall macro-average across all 45 subjects
    # ─────────────────────────────────────────────────────────────────────────
    per_category = {"STEM": [], "Applied Science": [], "HUMSS": [], "Other": []}
    for subject, info in results.items():
        per_category[KMMLU_CATEGORIES[subject]].append(info["accuracy"])

    category_avg = {cat: sum(scores) / len(scores) for cat, scores in per_category.items()}
    overall_avg = sum(info["accuracy"] for info in results.values()) / len(results)

    peak_vram_allocated_gb = torch.cuda.max_memory_allocated(device) / (1024 ** 3) if device == "cuda" else 0.0
    peak_vram_reserved_gb = torch.cuda.max_memory_reserved(device) / (1024 ** 3) if device == "cuda" else 0.0

    write_header = not os.path.exists(summary_csv)
    with open(summary_csv, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(
                ["Model", "Replacement", "Params", "Size (GB)",
                 "STEM", "Applied Science", "HUMSS", "Other", "Average",
                 "Peak VRAM Allocated (GB)", "Peak VRAM Reserved (GB)"]
            )
        writer.writerow(
            [
                model_label,
                args.replace_type or "",
                num_params,
                f"{model_size_gb:.2f}",
                f"{category_avg['STEM']:.4f}",
                f"{category_avg['Applied Science']:.4f}",
                f"{category_avg['HUMSS']:.4f}",
                f"{category_avg['Other']:.4f}",
                f"{overall_avg:.4f}",
                f"{peak_vram_allocated_gb:.2f}",
                f"{peak_vram_reserved_gb:.2f}",
            ]
        )

    print(f"\nParams: {num_params:,} ({num_params / 1e9:.2f}B)  Size: {model_size_gb:.2f} GB\n"
          f"STEM: {category_avg['STEM']:.4f}  Applied Science: {category_avg['Applied Science']:.4f}  "
          f"HUMSS: {category_avg['HUMSS']:.4f}  Other: {category_avg['Other']:.4f}  Average: {overall_avg:.4f}\n"
          f"Peak VRAM: {peak_vram_allocated_gb:.2f} GB allocated / {peak_vram_reserved_gb:.2f} GB reserved")
    print(f"Summary appended to {summary_csv}")
