"""
There were certain inconsitencies in the use of ppl and likelihood in the code
Correct all results to accommodate for the same
"""

import glob
import json
import os
import torch

# get all files in "results/EleutherAI/*/*.json"
file_list = glob.glob("results/EleutherAI/pythia-410m/*.json")

'''
dict_keys(['ppl', 'k_min_probs_0.05', 'k_min_probs_0.1', 'k_min_probs_0.2', 'k_min_probs_0.3', 'k_min_probs_0.4', 'k_min_probs_0.5', 'k_min_probs_0.6', 'k_max_probs_0.05', 'k_max_probs_0.1', 'k_max_probs_0.2', 'k_max_probs_0.3', 'k_max_probs_0.4', 'k_max_probs_0.5', 'k_max_probs_0.6', 'zlib_ratio', 'ppl_ratio_synonym_substitution', 'ppl_diff_synonym_substitution', 'ppl_ratio_butter_fingers', 'ppl_diff_butter_fingers', 'ppl_ratio_random_deletion', 'ppl_diff_random_deletion', 'ppl_ratio_change_char_case', 'ppl_diff_change_char_case', 'ppl_ratio_whitespace_perturbation', 'ppl_diff_whitespace_perturbation', 'ppl_ratio_underscore_trick', 'ppl_diff_underscore_trick', 'ref_ppl_ratio_silo', 'ref_ppl_diff_silo', 'ref_ppl_ratio_tinystories-33M', 'ref_ppl_diff_tinystories-33M', 'ref_ppl_ratio_tinystories-1M', 'ref_ppl_diff_tinystories-1M', 'ref_ppl_ratio_phi-1_5', 'ref_ppl_diff_phi-1_5'])
'''



# iterate over all files
for file in file_list:
    with open(file, 'r') as f:
        metrics = json.load(f)
        ppl_list = torch.tensor(metrics['ppl'])
        loss_list = torch.log(ppl_list)
        keys = list(metrics.keys())
        for key in keys:
            if "ref_ppl_ratio" in key:
                current_ratio = torch.tensor(metrics[key]) # loss_list / ref_ppl
                ref_ppl = loss_list / current_ratio
                ppl_ratio = ppl_list / ref_ppl
                loss_ratio = torch.log(ref_ppl) / loss_list
                metrics[key] = ppl_ratio.tolist()
                metrics[key.replace("ppl", "loss")] = loss_ratio.tolist()
            elif "ref_ppl_diff" in key:
                current_diff = torch.tensor(metrics[key]) # loss_list - ref_ppl
                ref_ppl = loss_list - current_diff
                ppl_diff = ppl_list - ref_ppl
                loss_diff = torch.log(ref_ppl) - loss_list
                metrics[key] = ppl_diff.tolist()
                metrics[key.replace("ppl", "loss")] = loss_diff.tolist()
            elif "ppl_ratio" in key:
                current_ratio = torch.tensor(metrics[key])
                perturbation_loss = loss_list / current_ratio
                perturbation_ppl = torch.exp(perturbation_loss)
                ppl_ratio = ppl_list / perturbation_ppl
                loss_ratio = perturbation_loss / loss_list
                metrics[key] = ppl_ratio.tolist()
                metrics[key.replace("ppl", "loss")] = loss_ratio.tolist()
            elif "ppl_diff" in key:
                current_diff = torch.tensor(metrics[key])
                perturbation_loss = loss_list - current_diff
                perturbation_ppl = torch.exp(perturbation_loss)
                ppl_diff = ppl_list - perturbation_ppl
                loss_diff = perturbation_loss - loss_list
                metrics[key] = ppl_diff.tolist()
                metrics[key.replace("ppl", "loss")] = loss_diff.tolist()
        
        # save the new file at "new_results/EleutherAI/*/*.json"
        new_file = file.replace("results", "new_results")
        os.makedirs(os.path.dirname(new_file), exist_ok=True)
        with open(new_file, 'w') as f:
            json.dump(metrics, f)




