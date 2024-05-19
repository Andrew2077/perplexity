import torch
import gc
import json
import os
import argparse
from calc_perplexity import create_model, get_vocabs, loop_func

def get_vram_usage():
    free, total = torch.cuda.mem_get_info()
    free_after, total_after = torch.cuda.mem_get_info()
    vram_usage = total_after - free_after - (total - free)
    return vram_usage / (1024 ** 2)  # Convert to MB

def get_vram(): 
    free = torch.cuda.mem_get_info()[0] / 1024 ** 3 
    total = torch.cuda.mem_get_info()[1] / 1024 ** 3 
    return total, free

def get_free_vram():
    return torch.cuda.mem_get_info()[0] / 1024 ** 3 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="gpt2", help="model name")
    parser.add_argument("--min_range", type=int, default=0, help="min range")
    parser.add_argument("--max_range", type=int, default=-1, help="max range")
    parser.add_argument("--load_in", type=str, default="4bit", help="load in")
    parser.add_argument("--main_dir", type=str, default=None)

    args = parser.parse_args()
    
    v1 = get_free_vram()
    MODEL, TOKENIZE = create_model(args.model_name, "auto", args.load_in)
    VOCABS, VOCAB_SIZE = get_vocabs(TOKENIZE)
    
    v2 = get_free_vram()
    loop_func(MODEL, TOKENIZE, VOCABS, VOCAB_SIZE, args.model_name, args.min_range, args.max_range)
    v3 = get_free_vram()
    
    torch.cuda.empty_cache()
    gc.collect()

    proecss_vram = v2 - v3
    model_vram = v1 - v2 

    print(f"proecss vram: {proecss_vram:3f} Gb , Model size vram : {model_vram:3f} Gb")

    info = {
        "model_load": model_vram,
        "model_forward": proecss_vram,
        "vocab_size": VOCAB_SIZE
    }

    path = os.path.join("perplexity",'inference stats', args.model_name)
    os.makedirs(path, exist_ok=True)
    json_path = os.path.join(path, f"inference_stat.json")
    json.dump(info, open(json_path, "w", encoding="utf-8"), ensure_ascii=False, indent=4)

