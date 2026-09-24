#!/usr/bin/env bash

# ======== Slurm setting
#SBATCH -J search-pruning-layer
#SBATCH --output=search-pruning-layer_%j.out
#SBATCH --error=search-pruning-layer_%j.err
#SBATCH --time=12:00:00
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

# See args.py: ModelArguments + SearchArguments (no --output_dir here — this
# script only searches for the pruning range, it doesn't save a model).
python search_pruning_layer.py \
  --model_name meta-llama/Llama-3.1-8B \
  --layer_intervals 8