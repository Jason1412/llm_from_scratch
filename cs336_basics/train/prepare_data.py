import numpy as np
from tqdm import tqdm
from pathlib import Path
import argparse
import multiprocessing as mp
from cs336_basics.bpe.tokenizer_orig import Tokenizer

_tokenizer = None

def _init_worker(tokenizer):
    global _tokenizer
    _tokenizer = tokenizer

def _process_chunk(lines):
    return np.array(list(_tokenizer.encode_iterable(lines)), dtype=np.uint16)

def _chunk_reader(input_path, chunk_size=10000):
    with open(input_path, "r", encoding="utf-8") as f:
        chunk = []
        for line in f:
            chunk.append(line)
            if len(chunk) >= chunk_size:
                yield chunk
                chunk = []
        if chunk:
            yield chunk

def prepare(input_path, output_path, tokenizer, num_processes=None):
    print(f"Processing {input_path} -> {output_path}")
    
    if num_processes is None:
        num_processes = max(1, mp.cpu_count() - 1)
        
    with open(output_path, "wb") as f_out:
        with mp.Pool(processes=num_processes, initializer=_init_worker, initargs=(tokenizer,)) as pool:
            for tokens in tqdm(pool.imap(_process_chunk, _chunk_reader(input_path))):
                tokens.tofile(f_out)
            
        
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_input", type=str, default="data/TinyStoriesV2-GPT4-train.txt")
    parser.add_argument("--val_input", type=str, default="data/TinyStoriesV2-GPT4-valid.txt")
    parser.add_argument("--train_output", type=str, default="data/train.bin")
    parser.add_argument("--val_output", type=str, default="data/val.bin")
    parser.add_argument("--vocab", type=str, default="bpe/vocab.json")
    parser.add_argument("--merges", type=str, default="bpe/merges.txt")
    
    args = parser.parse_args()
    
    tokenizer = Tokenizer.from_files(args.vocab, args.merges, special_tokens=["<|endoftext|>"])
    
    if Path(args.train_input).exists():
        prepare(args.train_input, args.train_output, tokenizer)
    else:
        print(f"Warning: {args.train_input} does not exist")
        
    if Path(args.val_input).exists():
        prepare(args.val_input, args.val_output, tokenizer)
    else:
        print(f"Warning: {args.val_input} does not exist")
        
if __name__ == "__main__":
    main()