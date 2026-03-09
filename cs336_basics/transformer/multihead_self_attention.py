import torch
import torch.nn as nn
from cs336_basics.transformer.linear import Linear
from cs336_basics.transformer.scaled_dot_product_attention import scaled_dot_product_attention
from cs336_basics.transformer.rotary_positional_embedding import RotaryPositionalEmbedding
from einops import rearrange

class MultiHeadAttn(nn.Module):
    def __init__(self, d_model: int, 
                num_heads: int,
                use_rope: bool = False,
                theta: float = 10000.0,
                max_seq_len: int = 2048,
                device: torch.device | None = None,
                dtype: torch.dtype | None = None):

        super(MultiHeadAttn, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.d_v = d_model // num_heads

        self.W_q = Linear(d_model, d_model, device=device, dtype=dtype)
        self.W_k = Linear(d_model, d_model, device=device, dtype=dtype)
        self.W_v = Linear(d_model, d_model, device=device, dtype=dtype)
        self.W_o = Linear(d_model, d_model, device=device, dtype=dtype)

        self.use_rope = use_rope
        if use_rope:
            self.rope = RotaryPositionalEmbedding(
                theta=theta, 
                d_k=self.d_k, 
                max_seq_len=max_seq_len, 
                device=device
            )

    def _create_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device)).bool() # Shape = (seq_len, seq_len)
        return mask

    def forward(self, X: torch.Tensor, 
                token_positions: torch.Tensor | None = None) -> torch.Tensor:
        '''
        Input:
            X: (batch_size, n_q, d_model)
            token_positions: (batch_size, n_q)
        Output:
            output: (batch_size, n_q, d_model)
        '''
        
        batch_size, n_q, d_model = X.shape # n_q --- seq len for the query
        causal_mask = self._create_causal_mask(n_q, device=X.device)


        Q = self.W_q(X)
        K = self.W_k(X)
        V = self.W_v(X)

        Q = rearrange(Q, 'b n (h d) -> b h n d', h=self.num_heads)
        K = rearrange(K, 'b n (h d) -> b h n d', h=self.num_heads)
        V = rearrange(V, 'b n (h d) -> b h n d', h=self.num_heads)

        if self.use_rope:
            if token_positions is None:
                token_positions = torch.arange(n_q, device=X.device).unsqueeze(0)
            Q = self.rope(Q, token_positions)
            K = self.rope(K, token_positions)

        attn = scaled_dot_product_attention(Q, K, V, mask=causal_mask)

        attn = rearrange(attn, 'b h n d -> b n (h d)')

        output = self.W_o(attn)

        return output
