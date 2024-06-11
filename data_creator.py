'''
This file is used to convert data from the PILE to a huggingface dataset.
This file will also call various perturbations, and add perturbed versions of the data to the dataset as different subsets.
'''
from dataloader import load_data, pile_mapper
from transform import generate_perturbations
import os
import json


def main(args):
    root = os.getcwd() + "/data"
    os.makedirs(root, exist_ok=True)

    if args.dataset_names == "all":
        dataset_names = pile_mapper.keys()
    else:
        dataset_names = [args.dataset_names]

    for dataset_name in dataset_names:
        for split in ["train", "val"]:
            file_name = f"{root}/{dataset_name}_{split}.jsonl"
            # load the data
            num_samples = 2000
            raw_texts = load_data(dataset_name, split, num_samples)
            print(f"Data loaded for {dataset_name} {split} | {len(raw_texts)} samples")
            # add the perturbations
            perturbed_texts_dictionary = generate_perturbations(raw_texts)
            perturbation_styles = list(perturbed_texts_dictionary.keys())
            
            #save all the texts to a json lines file
            with open(file_name, "w") as f:
                for i, text in enumerate(raw_texts):
                    json_line = {}
                    json_line["text"] = text
                    for style in perturbation_styles:
                        json_line[style] = perturbed_texts_dictionary[style][i]
                    f.write(json.dumps(json_line) + "\n")
            print(f"Data saved to {file_name}")
                    

            
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_names", type=str, default="all")
    args = parser.parse_args()
    
    main(args)
