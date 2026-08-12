#!/usr/bin/env bash

# ======== Slurm setting
#SBATCH -J benchmark-kmmlu
#SBATCH --output=benchmark-kmmlu_%j.out
#SBATCH --error=benchmark-kmmlu_%j.err
#SBATCH --time=4-00:00:00
#SBATCH -D /dss/dsshome1/07/di54rur/kim_he/LLM-VLM-in-Jetson/llm-streamline

# ======== General resourece setting
#SBATCH --mem=512gb
#SBATCH --clusters=hpda2
#SBATCH --partition=hpda2_compute_gpu
#SBATCH --cpus-per-task=4

# ======== GPU resource setting
#SBATCH --gres=gpu:1

# load required modules
module load slurm_setup
eval "$(micromamba shell hook --shell bash)"
micromamba activate llm-streamline
export PYTHONNOUSERSITE=1
export HYDRA_FULL_ERROR=1
export HF_HOME=/dss/dsstbyfs02/scratch/07/di54rur/.huggingface


cd /dss/dsshome1/07/di54rur/kim_he/LLM-VLM-in-Jetson/llm-streamline

# Each row of this table is one benchmark.py run; the per-subject CSV lets a
# requeued job resume mid-model instead of re-scoring finished subjects, and
# every run appends one row to <output_dir>/kmmlu_summary.csv.
# checkpoint dir can be a local pruned checkpoint OR a plain HF Hub id (e.g.
# facebook/opt-6.7b for the unpruned baseline) — from_pretrained downloads and
# caches Hub ids under $HF_HOME automatically, same as the other scripts do.

  # checkpoint dir                 | label                    | replace_type
# declare -a RUNS=(
#   "facebook/opt-6.7b| opt_6.7b| baseline"
#   "/dss/dsstbyfs02/scratch/07/di54rur/pseudolab/opt-6b-pruned/opt_prune_from2to9_none| opt_prune_from2to9_none| none"
#   "/dss/dsstbyfs02/scratch/07/di54rur/pseudolab/opt-6b-pruned/opt_prune_from2to9_mlp| opt_prune_from2to9_mlp| mlp"
#   "/dss/dsstbyfs02/scratch/07/di54rur/pseudolab/opt-6b-pruned/opt_prune_from2to9_tf| opt_prune_from2to9_tf| tf"
# )

# for run in "${RUNS[@]}"; do
#   IFS="|" read -r ckpt_dir label replace_type <<< "$run"
#   echo "=== Benchmarking ${label} (${replace_type}) : ${ckpt_dir} ==="
#   python benchmark.py \
#     --num_fewshot 1 \
#     --model_name "${ckpt_dir}" \
#     --model_label "${label}" \
#     --replace_type "${replace_type}" \
#     --output_dir /dss/dsstbyfs02/scratch/07/di54rur/pseudolab/opt-6b-pruned/benchmark_results
# done


# declare -a RUNS=(
#   "meta-llama/Llama-2-7b-hf| llama_2_7b| baseline"
#   "XiaodongChen/Llama-2-4.7B| llama_2_prune_(none)_tf| tf"
# )

# for run in "${RUNS[@]}"; do
#   IFS="|" read -r ckpt_dir label replace_type <<< "$run"
#   echo "=== Benchmarking ${label} (${replace_type}) : ${ckpt_dir} ==="
#   python benchmark.py \
#     --num_fewshot 5 \
#     --model_name "${ckpt_dir}" \
#     --model_label "${label}" \
#     --replace_type "${replace_type}" \
#     --output_dir /dss/dsstbyfs02/scratch/07/di54rur/pseudolab/opt-6b-pruned/benchmark_results
# done

# declare -a RUNS=(
#   "meta-llama/Llama-3.1-8B| llama_3_8b| baseline"
#   "XiaodongChen/Llama-3.1-5.4B| llama_3_prune_(none)_tf| tf"
# )

# for run in "${RUNS[@]}"; do
#   IFS="|" read -r ckpt_dir label replace_type <<< "$run"
#   echo "=== Benchmarking ${label} (${replace_type}) : ${ckpt_dir} ==="
#   python benchmark.py \
#     --num_fewshot 5 \
#     --model_name "${ckpt_dir}" \
#     --model_label "${label}" \
#     --replace_type "${replace_type}" \
#     --output_dir /dss/dsstbyfs02/scratch/07/di54rur/pseudolab/opt-6b-pruned/benchmark_results
# done




# ------------------------------------------
# VRAM Checking with a few examples
# ------------------------------------------
declare -a RUNS=(
  "facebook/opt-6.7b| opt_6.7b| baseline"
  "/dss/dsstbyfs02/scratch/07/di54rur/pseudolab/opt-6b-pruned/opt_prune_from2to9_none| opt_prune_from2to9_none| none"
  "/dss/dsstbyfs02/scratch/07/di54rur/pseudolab/opt-6b-pruned/opt_prune_from2to9_mlp| opt_prune_from2to9_mlp| mlp"
  "/dss/dsstbyfs02/scratch/07/di54rur/pseudolab/opt-6b-pruned/opt_prune_from2to9_tf| opt_prune_from2to9_tf| tf"
)

for run in "${RUNS[@]}"; do
  IFS="|" read -r ckpt_dir label replace_type <<< "$run"
  echo "=== Benchmarking ${label} (${replace_type}) : ${ckpt_dir} ==="
  python benchmark.py \
    --num_fewshot  1 \
    --max_examples 5 \
    --model_name "${ckpt_dir}" \
    --model_label "${label}_vramcheck" \
    --replace_type "${replace_type}" \
    --output_dir /dss/dsstbyfs02/scratch/07/di54rur/pseudolab/vram_check_results 
done

declare -a RUNS=(
  "meta-llama/Llama-2-7b-hf| llama_2_7b| baseline"
  "XiaodongChen/Llama-2-4.7B| llama_2_prune_(none)_tf| tf"
  "meta-llama/Llama-3.1-8B| llama_3_8b| baseline"
  "XiaodongChen/Llama-3.1-5.4B| llama_3_prune_(none)_tf| tf"
)

for run in "${RUNS[@]}"; do
  IFS="|" read -r ckpt_dir label replace_type <<< "$run"
  echo "=== Benchmarking ${label} (${replace_type}) : ${ckpt_dir} ==="
  python benchmark.py \
    --num_fewshot  5 \
    --max_examples 5 \
    --model_name "${ckpt_dir}" \
    --model_label "${label}_vramcheck" \
    --replace_type "${replace_type}" \
    --output_dir /dss/dsstbyfs02/scratch/07/di54rur/pseudolab/vram_check_results 
done