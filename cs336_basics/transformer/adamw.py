import torch
from torch.optim import Optimizer

class AdamW(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2):
        if lr < 0:
            raise ValueError("Invalid learning rate: {} - should be >= 0".format(lr))
        if betas[0] < 0 or betas[0] > 1:
            raise ValueError("Invalid beta parameter at index 0: {} - should be in [0, 1]".format(betas[0]))
        if betas[1] < 0 or betas[1] > 1:
            raise ValueError("Invalid beta parameter at index 1: {} - should be in [0, 1]".format(betas[1]))
        if eps < 0:
            raise ValueError("Invalid epsilon value: {} - should be >= 0".format(eps))
        if weight_decay < 0:
            raise ValueError("Invalid weight_decay value: {} - should be >= 0".format(weight_decay))

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super(AdamW, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            eps = group['eps']
            weight_decay = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                state['step'] += 1

                t = state['step']
                # updating the value of m
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

                # updating the value of v 
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                bias_correction1 = 1.0 - beta1 ** t
                bias_correction2 = 1.0 - beta2 ** t

                exp_avg_corrected = exp_avg / bias_correction1
                exp_avg_sq_corrected = exp_avg_sq / bias_correction2

                denom = exp_avg_sq_corrected.sqrt().add_(eps)

                if weight_decay != 0:
                    p.mul_(1 - lr * weight_decay)

                p.addcdiv_(exp_avg_corrected, denom, value=-lr)

        return loss
