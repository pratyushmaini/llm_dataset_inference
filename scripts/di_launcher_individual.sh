#!/bin/sh
#SBATCH --output=slurm_output/slurm_%j.out  # Standard output
#SBATCH --error=slurm_output/slurm_%j.err   # Standard error
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=A6000:1
#SBATCH --tasks-per-node=1
#SBATCH --mem=50G
#SBATCH --time=20:00:00
#SBATCH --mail-user=pratyus2@cs.cmu.edu
#SBATCH --partition=general

model_name=$1
split_name=$2
gpu_id=$3
batch_size=$4
dataset=$5

source ~/.bashrc
conda init

# if model_name is "kernelmachine/silo-pdswby-1.3b", then conda activate di_silo
# conda activate di

if [ $model_name = "kernelmachine/silo-pdswby-1.3b" ]
then
    conda activate di_silo
else
    conda activate di
fi

cd /home/pratyus2/projects/llm_dataset_inference

echo "model_name: $model_name" split_name: $split_name gpu_id: $gpu_id

echo "dataset: $dataset"
CUDA_VISIBLE_DEVICES=$gpu_id python di.py --split $split_name --dataset_name $dataset --model_name $model_name --batch_size $batch_size