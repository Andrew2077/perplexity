import time
start_time = time.time()


import subprocess
import concurrent.futures


def run_command(command):
    process = subprocess.Popen(command, shell=True)
    process.communicate()

model_name = "openai-community/gpt2"
MAX_LENGHT = -1


command1 = f'python calc_perplexity.py --model_name "{model_name}" --min_range 0 --max_range 1000'
command2 = f'python calc_perplexity.py --model_name "[GPT2MED]" --min_range 1000 --max_range 2000'
command3 = f'python calc_perplexity.py --model_name "GPT2MED" --min_range 2000 --max_range 3000'
# command2 = 'python calc_perplexity.py --model_name "GPT2MED" --min_range 200 --max_range 300'

with concurrent.futures.ThreadPoolExecutor() as executor:
    executor.map(run_command, [command1, command2, command3])
    
end_time = time.time()
total_time = end_time - start_time

print(f"Total execution time: {total_time} seconds")