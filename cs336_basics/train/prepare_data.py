import numpy as np
from tqdm import tqdm
from pathlib import Path
import argparse
import multiprocessing as mp
from cs336_basics.bpe.tokenizer_orig import Tokenizer

global_tokenizer = None

def prepare(input_path, output_path, tokenizer):
def init_worker(vocab_path, merges_path):
    global global_tokenizer
    global_tokenizer = Tokenizer.from_files(vocab_path, merges_path, special_tokens=["<|endoftext|>"])

def encode_chunk(lines):
    return list(global_tokenizer.encode_iterable(lines))

def prepare(input_path, output_path, vocab_path, merges_path):
    print(f"Processing {input_path} -> {output_path}")
    
    buffer = []
    buffer_size = 1024 * 1024
    num_processes = max(1, mp.cpu_count() - 1)
    
    with open(output_path, "wb") as f_out:
        with open(input_path, "r", encoding="utf-8") as f_in:
            for token_id in tqdm(tokenizer.encode_iterable(f_in)):
                buffer.append(token_id)
                
                if len(buffer) >= buffer_size:
                    np.array(buffer, dtype=np.uint16).tofile(f_out)
                    buffer = []
                    
                    
        if buffer:
            np.array(buffer, dtype=np.uint16).tofile(f_out)
            def chunk_generator():
                chunk = []
                for line in f_in:
                    chunk.append(line)
                    if len(chunk) >= 10000:
                        yield chunk
                        chunk = []
                if chunk:
                    yield chunk
            
            with mp.Pool(num_processes, initializer=init_worker, initargs=(vocab_path, merges_path)) as pool:
                for token_ids in tqdm(pool.imap(encode_chunk, chunk_generator())):
                    if token_ids:
                        np.array(token_ids, dtype=np.uint16).tofile(f_out)
            
        
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
        prepare(args.train_input, args.train_output, args.vocab, args.merges)
    else:
        print(f"Warning: {args.train_input} does not exist")
        
    if Path(args.val_input).exists():
        prepare(args.val_input, args.val_output, tokenizer)
        prepare(args.val_input, args.val_output, args.vocab, args.merges)
    else:
        print(f"Warning: {args.val_input} does not exist")
        
if __name__ == "__main__":
    main()
