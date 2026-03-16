import math

def learning_rate_schedule(
    it: int, # current iteration
    min_learning_rate: float,
    max_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):
    if it < warmup_iters:
        return it / warmup_iters * max_learning_rate
    elif it <= cosine_cycle_iters:
        input_cos = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters) * math.pi
        return min_learning_rate + (max_learning_rate - min_learning_rate) * 0.5 * (1 + math.cos(input_cos))
    else:
        return min_learning_rate
