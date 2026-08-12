#!/usr/bin/env bash

# ======== Slurm setting
#SBATCH -J replace-and-retrain
#SBATCH --output=replace-and-retrain_%j.out
#SBATCH --error=replace-and-retrain_%j.err
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
export MASTER_PORT=$((12000 + SLURM_JOB_ID % 20000))
export HYDRA_FULL_ERROR=1
export HF_HOME=/dss/dsstbyfs02/scratch/07/di54rur/.huggingface
export HF_HUB_ETAG_TIMEOUT=60

cd /dss/dsshome1/07/di54rur/kim_he/LLM-VLM-in-Jetson/llm-streamline

# See args.py: ModelArguments + TrainingArguments.
# --pruning_start_layer/--pruning_end_layer come from search_pruning_layer.py's
# printed "pruning_layer: [start, end]" output — update these before submitting.

# model_name                | model       | pruning_start_layer | pruning_end_layer
# facebook/opt-6.7b         | opt         | 2                   | 9
# meta-llama/Llama-3.1-8B   | llama       | 22                  | 29
# meta-llama/Llama-2-7b-hf  | llama       | 21                  | 28

# python replace_and_retrain.py \
#   --model_name facebook/opt-6.7b \
#   --model opt \
#   --replace tf \
#   --pruning_start_layer 2 \
#   --pruning_end_layer 9 \
#   --output_dir /dss/dsstbyfs02/scratch/07/di54rur/pseudolab/opt-6b-pruned \
#   --patience 5 \

python replace_and_retrain.py \
  --model_name meta-llama/Llama-3.1-8B \
  --model llama \
  --replace mlp \
  --pruning_start_layer 22 \
  --pruning_end_layer 29 \
  --output_dir /dss/dsstbyfs02/scratch/07/di54rur/pseudolab/llama-3.1-8B-pruned \
  --patience 5 \

# python replace_and_retrain.py \
#   --model_name meta-llama/Llama-2-7b-hf \
#   --model llama \
#   --replace tf \
#   --pruning_start_layer 21 \
#   --pruning_end_layer 28 \
#   --output_dir /dss/dsstbyfs02/scratch/07/di54rur/pseudolab/llama-2-7b-hf-pruned \
#   --patience 5 \