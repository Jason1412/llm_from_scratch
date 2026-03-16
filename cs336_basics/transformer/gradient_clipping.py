from collections.abc import Iterable
import torch

def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float, eps=1e-6):
    total_norm_sq = torch.tensor(0.0)
    first_grad = True

    for p in parameters:
        if p.grad is not None:
            if first_grad:
                total_norm_sq = total_norm_sq.to(p.grad.device)
                first_grad = False

            param_norm = torch.norm(p.grad.detach(), 2)
            total_norm_sq += param_norm ** 2

    total_norm = total_norm_sq.sqrt()

    clip_coef = max_l2_norm / (total_norm + eps)

    if clip_coef < 1.0:
        for p in parameters:
            if p.grad is not None:
                p.grad.detach().mul_(clip_coef)

    return total_norm