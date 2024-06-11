from transformers import AutoModelForCausalLM, AutoTokenizer
def prepare_model(model_name, cache_dir):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # pad token
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    tokenizer.model_max_length = 512

    model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir= cache_dir, trust_remote_code=True).cuda()
    print("Model loaded")
    return model, tokenizer