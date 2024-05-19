
import time
start_time = time.time()

from calc_perplexity import create_model, get_vocabs, loop_func
import concurrent.futures

MODEL, TOKENIZE = create_model("openai-community/gpt2", "cuda")
VOCABS, VOCAB_SIZE = get_vocabs(TOKENIZE)

tuple1 = ("openai-community/gpt2", 0, 1000)
tuple2 = ("openai-community/gpt2", 1000, 2000)
tuple3 = ("openai-community/gpt2", 2000, 3000)


def process(tuple_vals):
    loop_func(MODEL, TOKENIZE, VOCABS, VOCAB_SIZE, tuple_vals[0], tuple_vals[1], tuple_vals[2])


with concurrent.futures.ThreadPoolExecutor() as executor:
    executor.map(process, [tuple1, tuple2, tuple3, tuple3])

end_time = time.time()
total_time = end_time - start_time

print(f"Total execution time: {total_time} seconds")