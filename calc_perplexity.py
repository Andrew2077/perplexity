import json
import argparse
import os
import gc
import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

CELOSS = CrossEntropyLoss()

def clear_cache():
    torch.cuda.empty_cache()
    gc.collect()

def create_model(model_id: str, device: str = "auto", load_in: str = "float16"):
    if load_in == "8bit":
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=quantization_config)

    elif load_in=="4bit":
        quantization_config = BitsAndBytesConfig(load_in_4bit=True)
        model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=quantization_config)

    else:
        try:
            model = AutoModelForCausalLM.from_pretrained(model_id, cache_dir='models',torch_dtype=torch.float16,  device_map=device)
        except Exception as e:
            print(e)
            clear_cache()
            try:
                model = AutoModelForCausalLM.from_pretrained(model_id, cache_dir='models',  device_map=device)
            except Exception:
                clear_cache()
                model = AutoModelForCausalLM.from_pretrained(model_id, cache_dir='models',  device_map='cuda')
                
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True, cache_dir='models')
    return model, tokenizer

def get_vocabs(tokenizer):
    vocabs = {v:k for k,v in tokenizer.get_vocab().items()}
    TOTLA_VOCAB_SIZE = tokenizer.vocab_size
    print(f"vocab size: {TOTLA_VOCAB_SIZE}") # 50257
    return vocabs, TOTLA_VOCAB_SIZE

def forward_ultra_fast(model, tokenizer, token_id: int):
    if isinstance(token_id, list):
        token_id = token_id[0]
        
    decoded_text = tokenizer.decode(token_id)
    inputs = torch.tensor(token_id, device=model.device).unsqueeze(0).unsqueeze(0)
    labels = inputs.clone()

    with torch.no_grad():
        outputs = model(inputs)
        lm_logits = outputs.logits
        loss = F.cross_entropy(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1), reduction='none')
        ppl = torch.exp(loss)

    try:
        token_info = {
            "decoded_token": decoded_text,
            "loss": loss.item(),
            "ppl": ppl.item()
        }
    except Exception:
        return  {
            "decoded_token": decoded_text,
            "loss": None,
            "ppl": None
        }
    return token_info

 

def loop_func(model, tokenizer, vocabs: dict, vocabs_length:int, model_name:str, min_range:int, max_range:int, main_dir: str=None):
    import tqdm
    model_name = model_name.replace('/', '-')
    if max_range > vocabs_length or max_range == -1:
        max_range = vocabs_length
    
    if min_range < 0:
        min_range = 0
    
    vocabs_info = {}
    for i in tqdm.tqdm(range(min_range, max_range), desc="looping through vocab", total=max_range-min_range):
        token_info = forward_ultra_fast(model, tokenizer, i)
        vocabs_info[vocabs[i]] = token_info
    
    if main_dir == None:
        main_dir = os.path.join(os.getcwd(), "output")
    
    path = os.path.join(main_dir,"perplexity",'models', model_name)
    os.makedirs(path, exist_ok=True)
    json_path = os.path.join(path, f"vocabs_info_{min_range}_{max_range}.json")
    json.dump(vocabs_info, open(json_path, "w", encoding="utf-8"), ensure_ascii=False, indent=4)
    return vocabs_info

def multi_loop_func(model, tokenizer, vocabs: dict, vocabs_length:int, model_name:str, min_range:int, max_range:int, main_dir: str=None, split_size:int = 1000):
    all_slits = [i for i in range(min_range, max_range, split_size)]
    all_slits.append(max_range)
    for i in range(len(all_slits)-1):
        saved_model_name = model_name.replace('/', '-')
        path = os.path.join(main_dir,"perplexity",'models', saved_model_name, f"vocabs_info_{all_slits[i]}_{all_slits[i+1]}.json")
        if os.path.exists(path):
            print(f'skipping {all_slits[i] } to {all_slits[i+1] }')
            continue
        else:
            print(f'processing {all_slits[i] } to {all_slits[i+1] }')
            loop_func(model, tokenizer, vocabs, vocabs_length, model_name, all_slits[i], all_slits[i+1], main_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="gpt2", help="model name")
    parser.add_argument("--min_range", type=int, default=0, help="min range")
    parser.add_argument("--max_range", type=int, default=-1, help="max range")
    parser.add_argument("--load_in", type=str, default="float16")
    parser.add_argument("--main_dir", type=str, default=None)
    parser.add_argument("--split_size", type=int, default=None)
    args = parser.parse_args()
    MODEL, TOKENIZE = create_model(args.model_name, "auto", args.load_in)
    VOCABS, VOCAB_SIZE = get_vocabs(TOKENIZE)
    if args.split_size:
        multi_loop_func(MODEL, TOKENIZE, VOCABS, VOCAB_SIZE, args.model_name, args.min_range, args.max_range, main_dir=args.main_dir, split_size=args.split_size)
    else:
        loop_func(MODEL, TOKENIZE, VOCABS, VOCAB_SIZE, args.model_name, args.min_range, args.max_range, main_dir=args.main_dir)
