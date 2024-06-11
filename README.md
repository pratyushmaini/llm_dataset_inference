# LLM Dataset Inference: Detect Datasets, Not Strings

This repository contains data from 22 different domains of the PILE, divided into train and val sets. The data is in the form of a JSON file, with each entry containing the raw text, as well as various kinds of perturbations applied to it. The dataset is used to facilitate privacy research in language models, where the perturbed data can be used as reference detect the presence of a particular dataset in the training data of a language model.

## Quick Links

- [**arXiv Paper**](): Detailed information about the Dataset Inference V2 project, including the dataset, results, and additional resources.
- [**GitHub Repository**](): Access the source code, evaluation scripts, and additional resources for Dataset Inference.
- [**Dataset on Hugging Face**](https://huggingface.co/datasets/pratyushmaini/llm_dataset_inference): Direct link to download the various versons of the PILE dataset.
- [**Summary on Twitter**](): A concise summary and key takeaways from the project.


## Applicability 🚀

The dataset is in text format and can be loaded using the Hugging Face `datasets` library. It can be used to evaluate any causal or masked language model for the presence of specific datasets in its training pool. The dataset is *not* intended for direct use in training models, but rather for evaluating the privacy of language models. Please keep the validation sets, and the perturbed train sets private, and do not use them for training models.

## Loading the Dataset

To load the dataset, use the following code:

```python
from datasets import load_dataset
dataset = load_dataset("pratyushmaini/llm_dataset_inference", subset = "wikipedia", split = "train")
```

### Available perturbations:
<!-- ["synonym_substitution", "butter_fingers", "random_deletion", "change_char_case", "whitespace_perturbation", "underscore_trick"] -->
We use the NL-Augmenter library to apply the following perturbations to the data:
- `synonym_substitution`: Synonym substitution of words in the sentence.
- `butter_fingers`: Randomly changing characters from the sentence.
- `random_deletion`: Randomly deleting words from the sentence.
- `change_char_case`: Randomly changing the case of characters in the sentence.
- `whitespace_perturbation`: Randomly adding or removing whitespace from the sentence.
- `underscore_trick`: Adding underscores to the sentence.


## Citing Our Work

If you find our codebase and dataset beneficial, please cite our work:
```
@misc{mainidi2024,
      title={LLM Dataset Inference: Did you train on my dataset?}, 
      author={Pratyush Maini and Hengrui Jia and Nicolas Papernot and Adam Dziedzic},
      year={2024},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
