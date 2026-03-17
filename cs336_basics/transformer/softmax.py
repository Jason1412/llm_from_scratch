import torch


def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    '''
    Args:
        x: Input tensor
        dim: the dimension to perform softmax

    NOTE:
        - The shape of input and output will be identical.
    '''

    max_val = torch.max(x, dim=dim, keepdim=True).values

    exp_tensor = torch.exp(x - max_val)
    
    sum_exp = torch.sum(exp_tensor, dim=dim, keepdim=True)

    return exp_tensor / sum_exp