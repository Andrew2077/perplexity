import argparse
import time
import subprocess
import concurrent.futures

def run_command(command):
    process = subprocess.Popen(command, shell=True)
    process.communicate()

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--model_name", type=str, help="model name")
    argparser.add_argument("--load_in", type=str, default="4bit")
    argparser.add_argument("--total_size", type=int)
    argparser.add_argument("--splits", type=int)
    argparser.add_argument("--split_size", type=int)
    argparser.add_argument("--output_dir", type=str)
    args = argparser.parse_args()

    start_time = time.time()

    TOTAL_SIZE = args.total_size #256000
    SPLITS = args.splits #4
    SPLIT_SIZE = args.split_size #1000
    PART = TOTAL_SIZE // SPLITS

    command_list = []
    for split in range(SPLITS):
        model_name = f"{args.model_name}_{split}"
        command = f'python calc_perplexity.py --model_name "{model_name}" --min_range {split*PART} --max_range {(split+1)*PART} --split_size {args.split_size} --load_in "{args.load_in}" "{args.output_dir}"'
        command_list.append(command)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        executor.map(run_command, command_list)
        
    end_time = time.time()
    total_time = end_time - start_time

    print(f"Total execution time: {total_time} seconds")