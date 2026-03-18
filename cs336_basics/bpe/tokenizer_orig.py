import regex
from collections import defaultdict
from typing import Iterable, Iterator, List, Set, Tuple
import torch
import json
from tests.common import gpt2_bytes_to_unicode
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

TokenPair = tuple[bytes, bytes]


class Tokenizer:
    def __init__(self, vocab, merges, special_tokens=None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or []
        self.merges_priority_map = {pair: i for i, pair in enumerate(merges)}
        self.bytes_to_id = {v: k for k, v in vocab.items()}

        self.special_to_id = {}
        for token in self.special_tokens:
            token_bytes = token.encode('utf-8')

            for id_val, bytes_val in self.vocab.items():
                if bytes_val == token_bytes:
                    self.special_to_id[token] = id_val
                    break
        
        self.pat_compiled = regex.compile(PAT)
        self.special_pat_compiled = None
        if self.special_tokens:
            sorted_special_tokens = sorted(self.special_tokens, key=len, reverse=True)
            self.special_pat_compiled = regex.compile('|'.join(map(regex.escape, sorted_special_tokens)))

        
    def _get_bpe_merges(self, piece: bytes) -> List[bytes]:
        """
        Given a piece of bytes (not containing special tokens), output a list of bytes after 
        applying the merging rules.
        """

        parts = [bytes([b]) for b in piece]

        while len(parts) > 1:
            pairs = set()

            for i in range(len(parts) - 1):
                pair = (parts[i], parts[i + 1])
                if pair in self.merges_priority_map:
                    pairs.add(pair)
                    
            if not pairs:
                break

            best_pair = min(pairs, key=lambda pair: self.merges_priority_map[pair])

            new_parts = []
            i = 0
            while i < len(parts):
                if i < len(parts) - 1 and parts[i] == best_pair[0] and parts[i+1] == best_pair[1]:
                    new_parts.append(best_pair[0] + best_pair[1])
                    i += 2
                else:
                    new_parts.append(parts[i])
                    i += 1

            parts = new_parts

        return parts
    
    def _process_chunk(self, chunk: str) -> Iterator[int]:
        for match in self.pat_compiled.finditer(chunk):
            word = match.group()
            if not word:
                continue

            merged_pieces = self._get_bpe_merges(word.encode('utf-8'))
            
            for piece in merged_pieces:
                if piece in self.bytes_to_id:
                    yield self.bytes_to_id[piece]
                else:
                    if '<unk>' in self.special_to_id:
                        yield self.special_to_id['<unk>']
                    else:
                        yield self.special_to_id.get(self.special_tokens[0], 0)

    def _encode_generator(self, text: str) -> Iterator[int]:
        if not text:
            return

        if self.special_pat_compiled:
            last_end = 0
            for match in self.special_pat_compiled.finditer(text):
                start, end = match.span()
                if start > last_end:
                    yield from self._process_chunk(text[last_end:start])
                
                chunk = match.group()
                if chunk in self.special_to_id:
                    yield self.special_to_id[chunk]
                else:
                    chunk_bytes = chunk.encode('utf-8')
                    if chunk_bytes in self.bytes_to_id:
                        yield self.bytes_to_id[chunk_bytes]
                    else:
                        if '<unk>' in self.special_to_id:
                            yield self.special_to_id['<unk>']
                        else:
                            yield self.special_to_id.get(self.special_tokens[0], 0)
                last_end = end
            
            if last_end < len(text):
                yield from self._process_chunk(text[last_end:])
        else:
            yield from self._process_chunk(text)
    
    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: List[str] | None = None,
    ):
        '''
        - In the file of merges, each line is a pair of token separated by space.
        - In the file of vocab, it is a dict with the key is the token, value is the int index. 
        '''
        
        with open(merges_filepath, "r") as merges_file:
            merges: list[TokenPair]
            gpt2_byte_decoder = {v: k for k, v in gpt2_bytes_to_unicode().items()}
            merges_raw = [tuple(line.rstrip().split(" ")) for line in merges_file]
            merges = [
                (
                    bytes([gpt2_byte_decoder[token] for token in merge_token_1]),
                    bytes([gpt2_byte_decoder[token] for token in merge_token_2]),
                )
                for merge_token_1, merge_token_2 in merges_raw]

        with open(vocab_filepath, encoding="utf-8") as f:
            vocab_reversed = json.load(f)
            vocab = {
                gpt2_vocab_index: bytes(
                    [gpt2_byte_decoder[token] for token in gpt2_vocab_item]
                )
                for gpt2_vocab_item, gpt2_vocab_index in vocab_reversed.items()
            }

        return cls(vocab=vocab, merges=merges, special_tokens=special_tokens)
    


    def encode(self, text: str) -> List[int]:
        return list(self._encode_generator(text))
    
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for s in iterable:
            yield from self._encode_generator(s)

    def decode(self, ids: List[int]):
        
        tokens = []
        for id_val in ids:
            if id_val in self.vocab:
                tokens.append(self.vocab[id_val])
            else:
                if '<unk>' in self.special_to_id and self.special_to_id['<unk>'] in self.vocab:
                    tokens.append(self.vocab[self.special_to_id['<unk>']])
                else:
                    if self.special_tokens:
                        first_special = self.special_tokens[0].encode('utf-8')
                        tokens.append(first_special)
                    else:
                        tokens.append(b' ')


        all_bytes = b''.join(tokens)
        return all_bytes.decode('utf-8', errors='replace')



def main():
    vocab = {}
    merges = []

    for i in range(256):
        vocab[i] = bytes([i])

    next_id = 256

    special_tokens = ["<|endoftext|>", "<pad>", "<unk>"]

    for token in special_tokens:
        token_bytes = token.encode('utf-8')
        vocab[next_id] = token_bytes
        next_id += 1

    merges.append((b"h", b"i"))   # hi -> 256
    merges.append((b"t", b"h"))   # th -> 257
    merges.append((b"e", b"r"))   # er -> 258
    merges.append((b"th", b"e"))  # the -> 259

    vocab[next_id] = b"hi"; next_id += 1
    vocab[next_id] = b"th"; next_id += 1
    vocab[next_id] = b"er"; next_id += 1
    vocab[next_id] = b"the"; next_id += 1

    tokenizer = Tokenizer(vocab=vocab, merges=merges, special_tokens=special_tokens)


    text = "the tokenizer<|endoftext|>hi there!"

    ids = tokenizer.encode(text)
    print("编码后的ID序列:", ids)

    decoded_text = tokenizer.decode(ids)
    print("还原后的文本:", repr(decoded_text))

    print("还原是否正确:", decoded_text == text)



if __name__ == '__main__':
    main()