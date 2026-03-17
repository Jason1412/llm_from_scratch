from collections import Counter
import multiprocessing as mp

import regex as re
from typing import Iterator
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def _tokenize_parts(parts: list[str]) -> Counter:
    counts = Counter()
    for part in parts:
        for chunk in re.finditer(PAT, part):
            chunk_bytes = tuple(chunk.group().encode("utf-8"))
            counts[chunk_bytes] += 1
    return counts


def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str],
):
    
    vocab = {i: bytes([i]) for i in range(256)}

    for special_token in special_tokens:
        special_token_bytes = special_token.encode("utf-8")
        if special_token_bytes not in vocab.values():
            new_id = len(vocab)
            vocab[new_id] = special_token_bytes

    num_merges = vocab_size - len(vocab)


    if num_merges < 0:
        raise ValueError("Vocab size must be greater than the sum of 256 and the count of special tokens.")

    # Pre-tokenization, compute the count of each extracted chunk, easy to update the chunks
    with open(input_path, "r", encoding="utf-8", errors="replace") as f: 
        text = f.read() # text --- string, will encode and decode the file by utf-8, then output the content as text

    if special_tokens:
        sorted_special_tokens = sorted(special_tokens, key=len, reverse=True)
        special_pat = "|".join(re.escape(st) for st in sorted_special_tokens)
        parts = re.split(special_pat, text)
    else:
        parts = [text]

    try:
        num_cores = max(1, mp.cpu_count() - 1)
    except NotImplementedError:
        num_cores = 1

    # Safely chunk parts that are too large to ensure even distribution across processes
    safe_parts = []
    for part in parts:
        if len(part) > 1000000:  # Roughly 1MB threshold
            chunk_size = len(part) // num_cores
            idx = 0
            while idx < len(part):
                next_idx = idx + chunk_size
                if next_idx < len(part):
                    # Find a safe split point (newline followed by non-whitespace)
                    while next_idx < len(part) - 1 and not (part[next_idx] == '\n' and not part[next_idx + 1].isspace()):
                        next_idx += 1
                    next_idx += 1
                safe_parts.append(part[idx:next_idx])
                idx = next_idx
        else:
            safe_parts.append(part)

    batches = [safe_parts[i::num_cores] for i in range(num_cores)]
    counts = Counter()
    if num_cores > 1:
        with mp.Pool(num_cores) as pool:
            results = pool.map(_tokenize_parts, batches)
        for res in results:
            counts.update(res)
    else:
        counts = _tokenize_parts(safe_parts)

    merges = []

    # Pre-calculate pair counts once
    pair_counts = Counter()
    for sequence, freq in counts.items():
        for pair in zip(sequence, sequence[1:]):
            pair_counts[pair] += freq

    for _ in range(num_merges):
        if not pair_counts:
            break   

        best_pair = max(pair_counts, key=lambda p: (pair_counts[p], vocab[p[0]], vocab[p[1]]))
        new_id = len(vocab)

        vocab[new_id] = vocab[best_pair[0]] + vocab[best_pair[1]]
        merges.append((vocab[best_pair[0]], vocab[best_pair[1]]))

        new_counts = Counter()
        for sequence, freq in counts.items():
            if best_pair[0] not in sequence:
                new_counts[sequence] += freq
                continue

            new_sequence = []
            i = 0
            changed = False
            while i < len(sequence):
                if i < len(sequence) - 1 and sequence[i] == best_pair[0] and sequence[i+1] == best_pair[1]:
                    new_sequence.append(new_id)
                    i += 2
                    changed = True
                else:
                    new_sequence.append(sequence[i])
                    i += 1

            new_seq_tuple = tuple(new_sequence)
            new_counts[new_seq_tuple] += freq
            
            # Only update the counts of pairs if the sequence was actually merged
            if changed:
                for pair in zip(sequence, sequence[1:]):
                    pair_counts[pair] -= freq
                    if pair_counts[pair] <= 0:
                        del pair_counts[pair]
                for pair in zip(new_seq_tuple, new_seq_tuple[1:]):
                    pair_counts[pair] += freq

        counts = new_counts

    return vocab, merges
