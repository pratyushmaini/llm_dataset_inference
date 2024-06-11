import torch
import zlib
import tqdm, json 

loss_fct = torch.nn.CrossEntropyLoss(reduction="none")

def raw_values_batch(model, tokenizer, example_list):
    '''
    This function takes a list of strings and returns the loss values for each token in the string
    input:
        model: the language model
        tokenizer: the tokenizer
        example_list: a list of strings

    output:
        loss_list:  a list of lists. 
                    Each list contains the loss values for each token in the string

    '''
    max_length = tokenizer.model_max_length
    input_ids = tokenizer(example_list, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
    
    if model.device.type == "cuda":
        input_ids = {k: v.cuda() for k, v in input_ids.items()}
    
    # forward pass with no grad
    with torch.no_grad():
        outputs = model(**input_ids)
    
    labels = input_ids["input_ids"]
    labels[labels == tokenizer.pad_token_id] = -100

    # shift the labels
    shifted_labels = labels[..., 1:].contiguous().view(-1)

    # shift the logits
    shifted_logits = outputs.logits[..., :-1, :].contiguous()
    shifted_logits = shifted_logits.view(-1, shifted_logits.size(-1))

    loss = loss_fct(shifted_logits, shifted_labels)

    # reshape the loss to the original shape
    loss = loss.view(labels.size(0), labels.size(1) - 1)

    # now remove the 0 values and create loss as a list of lists
    loss_list = loss.tolist()
    
    for i,entry in enumerate(loss_list):
        # remove the 0 values
        entry = [x for x in entry if x != 0]
        loss_list[i] = entry
    
    # if any list is empty, remove it
    loss_list = [entry for entry in loss_list if len(entry) > 0]

    return loss_list

def raw_values(model, tokenizer, example_list, batch_size = 32):
    '''
    This function takes a list of strings and returns the loss values for each token in the string
    input:
        model: the language model
        tokenizer: the tokenizer
        example_list: a list of strings
        batch_size: the batch size
    output:
        loss_list:  a list of lists. 
                    Each list contains the loss values for each token in the string
    '''
    loss_list = []
    for i in tqdm.tqdm(range(0, len(example_list), batch_size)):
        batch = example_list[i:i + batch_size]
        loss_list += raw_values_batch(model, tokenizer, batch)
    return loss_list

def k_min_probs(loss_list, k=0.05, reverse=False):
    '''
    This function takes a list of lists and returns the ppl of the k fraction smallest values in each list
    input:
        loss_list: a list of lists
        k: the fraction of smallest values to return

    output:
        k_min_prob: the mean probability of the k fraction smallest values in each list
    '''
    # sort each list. if reverse is true, sort in reverse order (descending)
    sorted_list = [sorted(entry) for entry in loss_list]
    if reverse:
        sorted_list = [entry[::-1] for entry in sorted_list]
    k_min_probs = []
    for entry in sorted_list:
        # get the k fraction smallest values
        num_values = max(1, int(len(entry)*k))
        k_min = entry[:num_values]
        k_min_prob = sum(k_min)/len(k_min)
        k_min_probs.append(k_min_prob)
    return k_min_probs

def perplexity(loss_list):
    '''
    This function takes a list of lists and returns the perplexity of each list
    input:
        loss_list: a list of lists

    output:
        perplexity: the perplexity of each list
    '''
    perplexity = []
    for entry in loss_list:
        # calculate the mean of each list
        mean = sum(entry)/len(entry)
        # ppl is the exponent of the mean
        ppl = torch.exp(torch.tensor(mean)).item()
        perplexity.append(ppl)

    return perplexity

def zlib_ratio(loss_list, example_list):
    '''
    This function takes a list of lists and returns the ratio of the mean loss to the zlib compression of the input string
    input:
        loss_list: a list of lists
        example_list: a list of strings

    output:
        zlib_ratio: the ratio of the mean loss to the zlib compression of the input string
    '''
    zlib_ratios = []
    for i,entry in enumerate(loss_list):
        # calculate the mean of each list
        mean = sum(entry)/len(entry)
        # calculate the zlib compression of the input string
        zlib_entropy = len(zlib.compress(bytes(example_list[i], 'utf-8')))
        # calculate the ratio
        ratio = mean/zlib_entropy
        zlib_ratios.append(ratio)
    return zlib_ratios

def ppl_ratio(loss_list, reference_list):
    '''
    This function takes a list of lists and returns the ratio of the mean loss to the perplexity of a reference model
    input:
        loss_list: a list of lists
        reference_list: a list of perplexity values, or a list of lists of loss values

    output:
        ratio: the ratio of the mean loss to the perplexity of the reference model
    '''
    ratios = []
    for (entry, entry_ref) in zip(loss_list, reference_list):
        # calculate the mean of each list
        mean_model = sum(entry)/len(entry)
        if type(entry_ref) == list:
            mean_ref = sum(entry_ref)/len(entry_ref)
        else:
            mean_ref = entry_ref
        # calculate the ratio
        ratio = mean_model/mean_ref
        ratios.append(ratio)

    return ratios

def ppl_diff(loss_list, reference_list):
    '''
    This function takes a list of lists and returns the difference of the mean loss to the perplexity of a reference model
    input:
        loss_list: a list of lists
        reference_list: a list of perplexity values, or a list of lists of loss values

    output:
        diff: the difference of the mean loss to the perplexity of the reference model
    '''
    diffs = []
    for (entry, entry_ref) in zip(loss_list, reference_list):
        # calculate the mean of each list
        mean_model = sum(entry)/len(entry)
        if type(entry_ref) == list:
            mean_ref = sum(entry_ref)/len(entry_ref)
        else:
            mean_ref = entry_ref
        # calculate the ratio
        diff = mean_model - mean_ref
        diffs.append(diff)

    return diffs


def perturbation_ratio(model, tokenizer, dataset, loss_list, batch_size = 32):
    '''
    Dataset({
        features: ['text', 'synonym_substitution', 'butter_fingers', 'random_deletion', 'change_char_case', 'whitespace_perturbation', 'underscore_trick'],
        num_rows: 2000
    })
    '''
    result = {}
    for perturbation in dataset.column_names:
        if perturbation != "text":
            perturbed_list = dataset[perturbation]
            perturbed_loss_list = raw_values(model, tokenizer, perturbed_list, batch_size = batch_size)
            ratios = ppl_ratio(loss_list, perturbed_loss_list)
            diffs = ppl_diff(loss_list, perturbed_loss_list)
            result[f"ppl_ratio_{perturbation}"] = ratios
            result[f"ppl_diff_{perturbation}"] = diffs
    return result

    


reference_model_registry = {
    "silo":"kernelmachine/silo-pdswby-1.3b",
    "tinystories-33M": "roneneldan/TinyStories-33M",
    "tinystories-1M": "roneneldan/TinyStories-1M",
    "phi-1_5": "microsoft/phi-1_5",
    # "phi-1": "microsoft/phi-1",
}



def aggregate_metrics(model, tokenizer, dataset, metric_list, args, batch_size = 32):
    '''
    This function takes a list of strings and returns a dictionary of metrics
    input:
        model: the language model
        tokenizer: the tokenizer
        dataset: a huggingface dataset, with key "text" containing the strings
        metric_list: a list of metrics to calculate

    output:
        metrics: a dictionary of metrics
    '''
    example_list = dataset["text"]
    loss_list = raw_values(model, tokenizer, example_list, batch_size = batch_size)
    
    metrics = {}
    if "ppl" in metric_list:
        metrics["ppl"] = perplexity(loss_list)
    if "k_min_probs" in metric_list:
        for k in [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
            metrics[f"k_min_probs_{k}"] = k_min_probs(loss_list, k=k)
    if "k_max_probs" in metric_list:
        for k in [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
            metrics[f"k_max_probs_{k}"] = k_min_probs(loss_list, k=k, reverse=True)
    if "zlib_ratio" in metric_list:
        metrics["zlib_ratio"] = zlib_ratio(loss_list, example_list)

    if "perturbation" in metric_list:
        ratios_dict = perturbation_ratio(model, tokenizer, dataset, loss_list, batch_size)
        metrics.update(ratios_dict)

    if "reference_model" in metric_list:
        # for computation efficiency, we now enforce that the reference model should already have been run and its ppl saved
        for model_name in reference_model_registry:
            hf_path = reference_model_registry[model_name]
            with open(f"results/{hf_path}/{args.dataset_name}_{args.split}_metrics.json", 'r') as f:
                metrics_train = json.load(f)
                ref_ppl = metrics_train["ppl"]
                ref_ratios = ppl_ratio(loss_list, ref_ppl)
                ref_diffs = ppl_diff(loss_list, ref_ppl)
                metrics[f"ref_ppl_ratio_{model_name}"] = ref_ratios
                metrics[f"ref_ppl_diff_{model_name}"] = ref_diffs

        '''
        old code to run reference models on the fly
        from utils import prepare_model
        for model_name in reference_model_registry:
            hf_path = reference_model_registry[model_name]
            model, tokenizer = prepare_model(hf_path)
            
            reference_list = raw_values(model, tokenizer, example_list, batch_size = batch_size)
            metrics[f"ref_ppl_ratio_{model_name}"] = ppl_ratio(loss_list, reference_list)
            metrics[f"ref_ppl_diff_{model_name}"] = ppl_diff(loss_list, reference_list)
        '''

    return metrics