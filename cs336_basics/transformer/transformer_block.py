from torch import nn
import torch

from cs336_basics.transformer.rmsnorm import RMSNorm
from cs336_basics.transformer.multihead_self_attention import MultiHeadAttn
from cs336_basics.transformer.linear import Linear
from cs336_basics.transformer.positionwise_feedfoward import SwiGLUFFW


class TransformerBlock(nn.Module):
    def __init__(self, 
                d_model: int, 
                num_heads: int, 
                d_ff: int, 
                use_rope: bool = False,
                theta: float = 10000.0,
                max_seq_len: int = 2048,
                ):
        '''
        Args:
            d_model: dimension of the input to the transformer block.
            num_heads: number of heads to use in multi-head self-attention
            d_ff: dimension of the position-wise feed-forward inner layer
        '''
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.use_rope = use_rope
        self.theta = theta
        self.max_seq_len = max_seq_len

        self.rms_norm1 = RMSNorm(d_model)
        self.rms_norm2 = RMSNorm(d_model)
        self.attn = MultiHeadAttn(d_model, num_heads, use_rope, theta, max_seq_len)
        self.ffn = SwiGLUFFW(d_model, d_ff)


    def forward(self, 
                X: torch.Tensor,
                token_positions: torch.Tensor | None = None):
        '''
        Args:
            X: (batch_size, seq_len, d_model)
            token_positions: (batch_size, seq_len)
        '''
        x2 = self.rms_norm1(X)
        x2 = self.attn(x2, token_positions=token_positions)
        X = X + x2
        x2 = self.rms_norm2(X)
        x2 = self.ffn(x2)
        X = X + x2
        return X

