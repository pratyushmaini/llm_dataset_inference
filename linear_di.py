"""
Loads various features for the train and val sets.
Trains a linear model on the train set and evaluates it on the val set.

Tests p value of differentiating train versus val on held out features.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind, chi2, norm
import torch
import torch.nn as nn
import argparse
from tqdm import tqdm
from selected_features import feature_list

p_sample_list = [2, 5, 10, 20, 50, 100, 150, 200, 300, 400, 500, 600, 700, 800, 900, 1000]

def get_args():
    parser = argparse.ArgumentParser(description='Dataset Inference on a language model')
    parser.add_argument('--model_name', type=str, default="EleutherAI/pythia-12b", help='The name of the model to use')
    parser.add_argument('--dataset_name', type=str, default="wikipedia", help='The name of the dataset to use')
    parser.add_argument('--num_samples', type=int, default=1000, help='The number of samples to use')
    parser.add_argument("--normalize", type=str, default="train", help="Should you normalize?", choices=["no", "train", "combined"])
    parser.add_argument("--outliers", type=str, default="clip", help="The ablation to use", choices=["randomize", "keep", "zero", "mean", "clip", "mean+p-value", "p-value"])
    parser.add_argument("--features", type=str, default="all", help="The features to use", choices=["all", "selected"])
    parser.add_argument("--false_positive", type=int, default=0, help="What if you gave two val splits?", choices=[0, 1])
    parser.add_argument("--num_random", type=int, default=1, help="How many random runs to do?")
    args = parser.parse_args()
    return args


def get_model(num_features, linear = True):
    if linear:
        model = nn.Linear(num_features, 1)
    else:
        model = nn.Sequential(
            nn.Linear(num_features, 10),
            nn.ReLU(),
            nn.Linear(10, 1)  # Single output neuron
        )
    return model


def train_model(inputs, y, num_epochs=10000):
    num_features = inputs.shape[1]
    model = get_model(num_features)
        
    criterion = nn.BCEWithLogitsLoss()  # Binary Cross Entropy Loss for binary classification
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    # Convert y to float tensor for BCEWithLogitsLoss
    y_float = y.float()

    with tqdm(range(num_epochs)) as pbar:
        for epoch in pbar:
            optimizer.zero_grad()
            outputs = model(inputs).squeeze()  # Squeeze the output to remove singleton dimension
            loss = criterion(outputs, y_float)
            loss.backward()
            optimizer.step()
            pbar.set_description('loss {}'.format(loss.item()))
    return model

def get_predictions(model, val, y):
    with torch.no_grad():
        preds = model(val).detach().squeeze()
    criterion = nn.BCEWithLogitsLoss()
    loss = criterion(preds, y.float())
    return preds.numpy(), loss.item()

def get_dataset_splits(_train_metrics, _val_metrics, num_samples):
    # get the train and val sets
    for_train_train_metrics = _train_metrics[:num_samples]
    for_train_val_metrics = _val_metrics[:num_samples]
    for_val_train_metrics = _train_metrics[num_samples:]
    for_val_val_metrics = _val_metrics[num_samples:]


    # create the train and val sets
    train_x = np.concatenate((for_train_train_metrics, for_train_val_metrics), axis=0)
    train_y = np.concatenate((-1*np.zeros(for_train_train_metrics.shape[0]), np.ones(for_train_val_metrics.shape[0])))
    val_x = np.concatenate((for_val_train_metrics, for_val_val_metrics), axis=0)
    val_y = np.concatenate((-1*np.zeros(for_val_train_metrics.shape[0]), np.ones(for_val_val_metrics.shape[0])))
    
    # return tensors
    train_x = torch.tensor(train_x, dtype=torch.float32)
    train_y = torch.tensor(train_y, dtype=torch.float32)
    val_x = torch.tensor(val_x, dtype=torch.float32)
    val_y = torch.tensor(val_y, dtype=torch.float32)
    
    return (train_x, train_y), (val_x, val_y)

def normalize_and_stack(train_metrics, val_metrics, normalize="train"):
    '''
    excpects an input list of list of metrics
    normalize val with corre
    '''
    new_train_metrics = []
    new_val_metrics = []
    for (tm, vm) in zip(train_metrics, val_metrics):
        if normalize == "combined":
            combined_m = np.concatenate((tm, vm))
            mean_tm = np.mean(combined_m)
            std_tm = np.std(combined_m)
        else:
            mean_tm = np.mean(tm)
            std_tm = np.std(tm)
        
        if normalize == "no":
            normalized_vm = vm
            normalized_tm = tm
        else:
            #normalization should be done with respect to the train set statistics
            normalized_vm = (vm - mean_tm) / std_tm
            normalized_tm = (tm - mean_tm) / std_tm
        
        new_train_metrics.append(normalized_tm)
        new_val_metrics.append(normalized_vm)

    train_metrics = np.stack(new_train_metrics, axis=1)
    val_metrics = np.stack(new_val_metrics, axis=1)
    return train_metrics, val_metrics

def remove_outliers(metrics, remove_frac=0.05, outliers = "zero"):
    # Sort the array to work with ordered data
    sorted_ids = np.argsort(metrics)
    
    # Calculate the number of elements to remove from each side
    total_elements = len(metrics)
    elements_to_remove_each_side = int(total_elements * remove_frac / 2) 
    
    # Ensure we're not attempting to remove more elements than are present
    if elements_to_remove_each_side * 2 > total_elements:
        raise ValueError("remove_frac is too large, resulting in no elements left.")
    
    # Change the removed metrics to 0.
    lowest_ids = sorted_ids[:elements_to_remove_each_side]
    highest_ids = sorted_ids[-elements_to_remove_each_side:]
    all_ids = np.concatenate((lowest_ids, highest_ids))

    # import pdb; pdb.set_trace()
    
    trimmed_metrics = np.copy(metrics)
    
    if outliers == "zero":
        trimmed_metrics[all_ids] = 0
    elif outliers == "mean" or outliers == "mean+p-value":
        trimmed_metrics[all_ids] = np.mean(trimmed_metrics)
    elif outliers == "clip":
        highest_val_permissible = trimmed_metrics[highest_ids[0]]
        lowest_val_permissible = trimmed_metrics[lowest_ids[-1]]
        trimmed_metrics[highest_ids] =  highest_val_permissible
        trimmed_metrics[lowest_ids] =   lowest_val_permissible
    elif outliers == "randomize":
        #this will randomize the order of metrics
        trimmed_metrics = np.delete(trimmed_metrics, all_ids)
    else:
        assert outliers in ["keep", "p-value"]
        pass
        
    
    return trimmed_metrics
    

def get_p_value_list(heldout_train, heldout_val):
    p_value_list = []
    for num_samples in p_sample_list:
        heldout_train_curr = heldout_train[:num_samples]
        heldout_val_curr = heldout_val[:num_samples]
        t, p_value = ttest_ind(heldout_train_curr, heldout_val_curr, alternative='less')
        p_value_list.append(p_value)
    return p_value_list
    
    

def split_train_val(metrics):
    keys = list(metrics.keys())
    num_elements = len(metrics[keys[0]])
    print (f"Using {num_elements} elements")
    # select a random subset of val_metrics (50% of ids)
    ids_train = np.random.choice(num_elements, num_elements//2, replace=False)
    ids_val = np.array([i for i in range(num_elements) if i not in ids_train])
    new_metrics_train = {}
    new_metrics_val = {}
    for key in keys:
        new_metrics_train[key] = np.array(metrics[key])[ids_train]
        new_metrics_val[key] = np.array(metrics[key])[ids_val]
    return new_metrics_train, new_metrics_val

def main():
    args = get_args()
    with open(f"new_results/{args.model_name}/{args.dataset_name}_train_metrics.json", 'r') as f:
        metrics_train = json.load(f)
    with open(f"new_results/{args.model_name}/{args.dataset_name}_val_metrics.json", 'r') as f:
        metrics_val = json.load(f)

    if args.false_positive:
        metrics_train, metrics_val = split_train_val(metrics_val)

    keys = list(metrics_train.keys())
    train_metrics = []
    val_metrics = []
    for key in keys:
        if args.features != "all":
            if key not in feature_list:
                continue
        metrics_train_key = np.array(metrics_train[key])
        metrics_val_key = np.array(metrics_val[key])

        # remove the top 2.5% and bottom 2.5% of the data
        
        metrics_train_key = remove_outliers(metrics_train_key, remove_frac = 0.05, outliers = args.outliers)
        metrics_val_key = remove_outliers(metrics_val_key, remove_frac = 0.05, outliers = args.outliers)

        train_metrics.append(metrics_train_key)
        val_metrics.append(metrics_val_key)

    # concatenate the train and val metrics by stacking them
    
    # train_metrics, val_metrics = new_train_metrics, new_val_metrics
    train_metrics, val_metrics = normalize_and_stack(train_metrics, val_metrics)

    for i in range(args.num_random):
        np.random.shuffle(train_metrics)
        np.random.shuffle(val_metrics)
        
        # train a model by creating a train set and a held out set
        num_samples = args.num_samples 
        (train_x, train_y), (val_x, val_y) = get_dataset_splits(train_metrics, val_metrics, num_samples)
        
        model = train_model(train_x, train_y, num_epochs = 1000)
        preds, loss = get_predictions(model, val_x, val_y)
        preds_train, loss_train = get_predictions(model, train_x, train_y)
        og_train = preds_train[train_y == 0]
        og_val = preds_train[train_y == 1]

        heldout_train = preds[val_y == 0]
        heldout_val = preds[val_y == 1]
        # alternate hypothesis: heldout_train < heldout_val
        
        if args.outliers == "p-value" or args.outliers == "mean+p-value":
            heldout_train = remove_outliers(heldout_train, remove_frac = 0.05, outliers = "randomize")
            heldout_val = remove_outliers(heldout_val, remove_frac = 0.05, outliers = "randomize")

        p_value_list = get_p_value_list(heldout_train, heldout_val)

        # using the model weights, get importance of each feature, and save to csv
        weights = model.weight.data.squeeze().tolist() 
        features = keys
        feature_importance = {feature: weight for feature, weight in zip(features, weights)}
        df = pd.DataFrame(list(feature_importance.items()), columns=['Feature', 'Importance'])
        import os
        path_to_append = f"{args.outliers}-outliers/{args.normalize}-normalize"
        if args.features == "selected":
            path_to_append += "-selected_features"
        if args.false_positive:
            path_to_append += f"-{args.false_positive}-false_positive"

        model_name = args.model_name.replace("/", "_")
        os.makedirs(f"aggregated_results/feature_importance/{path_to_append}/{model_name}", exist_ok=True)
        df.to_csv(f'aggregated_results/feature_importance/{path_to_append}/{model_name}/{args.dataset_name}_seed_{i}.csv',  index=False)


        # add the  to the csv in p_values/{model_name}.csv if it does not exist
        os.makedirs(f"aggregated_results/p_values/{path_to_append}/{model_name}", exist_ok=True)
       
        p_file = f"aggregated_results/p_values/{path_to_append}/{model_name}/{args.dataset_name}.csv"
        print(f"Writing to {p_file}")
        if not os.path.exists(p_file):
            with open(p_file, 'w') as f:
                to_write = "seed," + ",".join([f"p_{str(p)}" for p in p_sample_list]) + "\n"
                f.write(to_write)
            
        # check if the dataset_name is already in the file
        flag = 0
        seed = f"seed_{i}"
        with open(p_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if seed in line:
                    print(f"Dataset {args.dataset_name} already in file {p_file}. Aborting...\n{p_value_list}")
                    flag = 1

            if flag == 0:
                with open(p_file, 'a') as f:
                    to_write = seed + "," + ",".join([str(p) for p in p_value_list]) + "\n"
                    f.write(to_write)

if __name__ == "__main__":
    main()