from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def prepare_model(model_name, cache_dir, quant=None):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # pad token
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    tokenizer.model_max_length = 512

    if quant is not None:
        model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir, trust_remote_code=True).cuda()
    elif quant == "fp16":
        model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir, trust_remote_code=True, torch_dtype=torch.bfloat16).cuda()
    elif quant == "8bit":
        model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir, trust_remote_code=True, load_in_8bit=True).cuda()

    print("Model loaded")
    return model, tokenizer