import numpy as np
from tqdm import tqdm
import argparse



def prepare(input_path, output_path, tokenizer):
    print(f"Processing {input_path} -> {output_path}")
    
    buffer = []
    buffer_size = 1024 * 1024
    
    with open(output_path, "wb") as f_out:
        with open(input_path, "r", encoding="utf-8") as f_in:
            for token_id in tqdm(tokenizer.encode_iterable(f_in)):
                buffer.append(token_id)
                
                if len(buffer) >= buffer_size:
                    np.array(buffer, dtype=np.uint16).tofile(f_out)
                    buffer = []
                    
                    
        if buffer:
            np.array(buffer, dtype=np.uint16).tofile(f_out)
            
        
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_input", type=str, default="data/TinyStoriesV2-GPT4-train.txt")
    parser.add_argument("--val_input", type=str, default="data/TinyStoriesV2-GPT4-valid.txt")
    parser.add_argument("--train_output", type=str, default="data/train.bin")
    parser.add_argument("--val_output", type=str, default="data/val.bin")
    parser.add_argument("--vocab", type=str, default="bpe/vocab.json")
    parser.add_argument("--merges", type=str, default="bpe/merges.txt")
    
    args = parser.parse_args()
    
    