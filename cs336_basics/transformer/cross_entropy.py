from torch import Tensor
import torch
from jaxtyping import Bool, Float, Int
from einops import reduce, rearrange


def cross_entropy(
        input_logits: Tensor, 
        targets: Tensor):
    '''
    Description:
        - This function computes the loss of a single predicted target.
    Args:
        - inputs: Shape[batch_size, vocab_size]
        - targets: Shape[batch_size], 
    '''
    max_logits, _ = torch.max(input_logits, dim=-1, keepdim=True)  # Find the max along vocab_size dimension
    shifted_logits = input_logits - max_logits
    
    logits_exp = shifted_logits.exp()

    log_exp_sum = torch.log(reduce(logits_exp, '... vocab_size -> ... 1', 'sum'))

    target_indices = rearrange(targets, '... -> ... 1')
    target_logits = shifted_logits.gather(dim=-1, index=target_indices)

    loss = log_exp_sum - target_logits

    return loss.mean()


