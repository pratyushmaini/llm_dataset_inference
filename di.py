from utils import prepare_model
from metrics import aggregate_metrics, reference_model_registry
import json, os
import argparse
from datasets import load_dataset

def get_args():
    parser = argparse.ArgumentParser(description='Dataset Inference on a language model')
    parser.add_argument('--model_name', type=str, default="EleutherAI/pythia-410m-deduped", help='The name of the model to use')
    parser.add_argument('--dataset_name', type=str, default="wikipedia", help='The name of the dataset to use')
    parser.add_argument('--split', type=str, default="train", help='The split of the dataset to use')
    parser.add_argument('--num_samples', type=int, default=1000, help='The number of samples to use')
    parser.add_argument('--batch_size', type=int, default=32, help='The batch size to use')
    parser.add_argument('--from_hf', type=int, default=1, help='If set, will load the dataset from huggingface')
    parser.add_argument('--cache_dir', type=str, default="/data/locus/llm_weights", help='The directory to cache the model')
    args = parser.parse_args()
    return args



def main():
    args = get_args()
    results_file = f"results/{args.model_name}/{args.dataset_name}_{args.split}_metrics.json"
    # if os.path.exists(results_file):
        # print(f"Results file {results_file} already exists. Aborting...")
        # return
    model_name =  args.model_name
    
    if model_name in ["microsoft/phi-1_5", "EleutherAI/pythia-12b", "EleutherAI/pythia-6.9b", "EleutherAI/pythia-410m"]:
        args.cache_dir = "/data/locus/llm_weights/pratyush"

    model, tokenizer = prepare_model(model_name, cache_dir= args.cache_dir)
    
    # load the data
    dataset_name = args.dataset_name
    split = args.split
    
    if not args.from_hf:
        from dataloader import load_data
        # if you want to load data directly from the PILE, use the following line
        num_samples = args.num_samples
        dataset = load_data(dataset_name, split, num_samples)
    else:
        dataset_path = f"data/{dataset_name}_{split}.jsonl"
        dataset = load_dataset("json", data_files=dataset_path, split="train")
    print("Data loaded")

    # get the metrics
    if model_name in reference_model_registry.values():
        metric_list = ["ppl"]
    else:
        metric_list = ["k_min_probs", "ppl", "zlib_ratio", "k_max_probs", "perturbation", "reference_model"]
    metrics = aggregate_metrics(model, tokenizer, dataset, metric_list, args, batch_size = args.batch_size)
    
    # save the metrics
    os.makedirs(f"results/{model_name}", exist_ok=True)
    with open(results_file, 'w') as f:
        json.dump(metrics, f)

if __name__ == "__main__":
    main()

