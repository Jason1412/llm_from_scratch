
from cs336_basics.transformer.transformer_lm import TransformerLM
from cs336_basics.transformer.softmax import softmax
import torch
import math

def decoding(
    lm: TransformerLM,
    inpput_ids: torch.Tensor, # shape (batch_size, seq_len)
    eos_id: int,
    maximum_tokens: int = 0,
    temperature: float = 0.7,
    top_p: float = 0.9
):
    count = 0
    while maximum_tokens == 0 or count < maximum_tokens:
        count += 1
        logits = lm(inpput_ids) # shape of logits: (batch_size, seq_len, vocab_size)
        
        if math.isclose(temperature, 0.0):
            next_token_index = torch.argmax(logits, dim=-1, keepdim=True)
        else:
            logits = logits / temperature
            
            if top_p < 1.0:
                next_token_logits = top_p_filter(logits, top_p)

            probs = softmax(next_token_logits, dim=-1)
            next_token_index = torch.multinomial(probs, num_samples=1)

        inpput_ids = torch.cat([inpput_ids, next_token_index], dim=-1)
        if torch.all(next_token_index == eos_id):
            break

    return idx


def top_p_filter(logits: torch.Tensor,
        top_p: float):
    '''
    Input Args:
        logits: (batch_size, vocab_size), because the next token is just one token.
        top_p: a threshold for the cumulative sum of probability.
    Output:
        revised_logits: (batch_size, vocab_size), with some elements set to be 0.
    '''
    sorted_logits, sorted_indices = torch.sort(logits, dim=-1, descending=True)
    cumulative_probs = torch.cumsum(softmax(sorted_logits, dim=-1), dim=-1)

    # Remove tokens with cumulative probability above the threshold
    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
    logits[indices_to_remove] = float('-inf')

    return logits
