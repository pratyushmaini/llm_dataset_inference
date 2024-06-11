#!/bin/sh
#SBATCH --output=slurm_output/slurm_%j.out  # Standard output
#SBATCH --error=slurm_output/slurm_%j.err   # Standard error
#SBATCH --cpus-per-task=4
#SBATCH --tasks-per-node=1
#SBATCH --mem=10G
#SBATCH --time=1-10:00:00
#SBATCH --partition=general
#SBATCH --array=0-999


datasets=(stackexchange wikipedia cc github pubmed_abstracts openwebtext2 freelaw math nih uspto hackernews enron books3 pubmed_central gutenberg arxiv bookcorpus2 opensubtitles youtubesubtitles ubuntu europarl philpapers)
outliers=("mean" "p-value")
normalizes=("combined" "train" "no")
features=("all" "selected")
false_positives=(1 0)
models=("EleutherAI/pythia-12b-deduped" "EleutherAI/pythia-12b" "EleutherAI/pythia-6.9b-deduped" "EleutherAI/pythia-6.9b" "EleutherAI/pythia-1.3b-deduped" "EleutherAI/pythia-1.3b" "EleutherAI/pythia-410m-deduped" "EleutherAI/pythia-410m")

#add 6000 to the array index to get the next model
SLURM_ARRAY_TASK_ID=$((SLURM_ARRAY_TASK_ID + 1000))

# Calculate the array index
dataset_idx=$((SLURM_ARRAY_TASK_ID % ${#datasets[@]}))
outlier_idx=$((SLURM_ARRAY_TASK_ID / ${#datasets[@]} % ${#outliers[@]}))
normalize_idx=$((SLURM_ARRAY_TASK_ID / (${#datasets[@]} * ${#outliers[@]}) % ${#normalizes[@]}))
feature_idx=$((SLURM_ARRAY_TASK_ID / (${#datasets[@]} * ${#outliers[@]} * ${#normalizes[@]}) % ${#features[@]}))
false_positive_idx=$((SLURM_ARRAY_TASK_ID / (${#datasets[@]} * ${#outliers[@]} * ${#normalizes[@]} * ${#features[@]}) % ${#false_positives[@]}))
model_idx=$((SLURM_ARRAY_TASK_ID / (${#datasets[@]} * ${#outliers[@]} * ${#normalizes[@]} * ${#features[@]} * ${#false_positives[@]})))

dataset=${datasets[$dataset_idx]}
outlier=${outliers[$outlier_idx]}
normalize=${normalizes[$normalize_idx]}
features=${features[$feature_idx]}
false_positive=${false_positives[$false_positive_idx]}
model_name=${models[$model_idx]}

if [ $false_positive -eq 1 ]; then
    num_samples=500
else
    num_samples=1000
fi

# model_name=$1
# outlier=$2
# normalize=$3
# features=$4
# false_positive=$5
# num_samples=$6
# dataset=$7

echo model_name: $model_name outliers: $outlier normalize: $normalize features: $features false_positive: $false_positive num_samples: $num_samples dataset: $dataset


source ~/.bashrc
conda init
conda activate di

cd /home/pratyus2/projects/llm_dataset_inference

python linear_di.py --num_random 10 --dataset_name $dataset --model_name $model_name --normalize $normalize --outliers $outlier --features $features --false_positive $false_positive --num_samples $num_samples
