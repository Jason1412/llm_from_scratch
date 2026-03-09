from einops import einsum
from cs336_basics.transformer.softmax import softmax
from jaxtyping import Bool, Float, Int
import torch
from torch import Tensor



def scaled_dot_product_attention(
    Q: Float[Tensor, "batch_size ... Lq d_k"],
    K: Float[Tensor, "batch_size ... Lk d_k"],
    V: Float[Tensor, "batch_size ... Lk d_v"],
    mask: Bool[Tensor, "... Lq Lk"] | None = None,
) -> Float[Tensor, "batch_size ... seq_len_queries d_v"]:
    
    d_k = Q.shape[-1]

    scale = torch.tensor(d_k, dtype=Q.dtype, device=Q.device).sqrt()

    attn_scores = einsum(Q, K, 'b ... Lq d_k, b ... Lk d_k -> b ... Lq Lk') / scale

    if mask is not None:
        attn_scores = attn_scores.masked_fill(mask==False, float('-inf'))


    weights = softmax(attn_scores, dim=-1)

    # print("Shape of scores =", attn_scores.shape)
    # print("Shape of weights =", weights.shape)
    # print("Shape of V =", V.shape)

    output = einsum(weights, V, 'b ... Lq Lk, b ... Lk d_v -> b ... Lq d_v')

    return output


# if __name__ == "__main__":
#     Q = torch.randn(10, 6, 3)
#     K = torch.randn(10, 4, 3)
#     V = torch.randn(10, 4, 5)
    
#     print(scaled_dot_product_attention(Q, K, V).shape)