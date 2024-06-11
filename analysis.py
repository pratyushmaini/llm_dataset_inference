'''
This file calculates p values by loading the json from results
'''
import json, os
import argparse
import numpy as np
from scipy.stats import ttest_ind, chi2, norm


def get_args():
    parser = argparse.ArgumentParser(description='Dataset Inference on a language model')
    parser.add_argument('--model_name', type=str, default="EleutherAI/pythia-12b", help='The name of the model to use')
    parser.add_argument('--dataset_name', type=str, default="wikipedia", help='The name of the dataset to use')
    parser.add_argument('--num_samples', type=int, default=1000, help='The number of samples to use')
    parser.add_argument('--batch_size', type=int, default=32, help='The batch size to use')
    args = parser.parse_args()
    return args

def fishers_method(p_values):
    statistic = -2 * np.sum(np.log(p_values))
    combined_p_value = chi2.sf(statistic, 2 * len(p_values))
    return combined_p_value

def harmonic_mean(p_values):
    return len(p_values) / np.sum(1. / np.array(p_values))

def get_p_values_averaged(list1, list2):
    # make 10 random samples of the two lists by sampling without replacement
    num_elements = min(len(list1), len(list2))
    num_elements_per_sample = int(num_elements/10)
    # randomly permute the two lists
    np.random.shuffle(list1)
    np.random.shuffle(list2)
    p_values = []
    for i in range(10):
        sample1 = list1[i*num_elements_per_sample:(i+1)*num_elements_per_sample]
        sample2 = list2[i*num_elements_per_sample:(i+1)*num_elements_per_sample]
        t_stat, p_value = ttest_ind(sample1, sample2)
        p_values.append(p_value)

    return harmonic_mean(p_values)

def get_p_values(list1, list2):
    t_stat, p_value = ttest_ind(list1, list2)
    return p_value

def main():
    args = get_args()
    with open(f"new_results/{args.model_name}/{args.dataset_name}_train_metrics.json", 'r') as f:
        metrics_train = json.load(f)
    with open(f"new_results/{args.model_name}/{args.dataset_name}_val_metrics.json", 'r') as f:
        metrics_val = json.load(f)

    keys = list(metrics_train.keys())
    p_values = {}
    for key in keys:
        # remove the top 2.5% and bottom 2.5% of the data
        metrics_train_key = np.array(metrics_train[key])
        metrics_val_key = np.array(metrics_val[key])
        metrics_train_key = metrics_train_key[np.argsort(metrics_train_key)]
        metrics_val_key = metrics_val_key[np.argsort(metrics_val_key)]
        metrics_train_key = metrics_train_key[int(0.025*len(metrics_train_key)):int(0.975*len(metrics_train_key))]
        metrics_val_key = metrics_val_key[int(0.025*len(metrics_val_key)):int(0.975*len(metrics_val_key))]
        # shuffle the data
        np.random.shuffle(metrics_train_key)
        np.random.shuffle(metrics_val_key)
        # get the p value
        # t_stat, p_value = ttest_ind(metrics_train_key, metrics_val_key)


        p_values[key] = get_p_values(metrics_train[key], metrics_val[key])
    
    # add the p_values to the csv in p_values_averaged/{args.model_name}/{key}.csv if it does not exist
    os.makedirs(f"p_values/{args.model_name}", exist_ok=True)
    for key in p_values:
        p_file = f"p_values/{args.model_name}/{key}.csv"
        if not os.path.exists(p_file):
            with open(p_file, 'w') as f:
                f.write("dataset_name,p_value\n")
        
        # check if the dataset_name is already in the file
        flag = 0
        with open(p_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if args.dataset_name in line:
                    print(f"Dataset {args.dataset_name} already in file {p_file}. Aborting...")
                    flag = 1

            if flag == 0:
                with open(p_file, 'a') as f:
                    f.write(f"{args.dataset_name},{p_values[key]}\n")

if __name__ == "__main__":
    main()