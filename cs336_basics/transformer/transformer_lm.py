from torch import nn
import torch

from cs336_basics.transformer.embedding import Embedding
from cs336_basics.transformer.transformer_block import TransformerBlock
from cs336_basics.transformer.rotary_positional_embedding import RotaryPositionalEmbedding
from cs336_basics.transformer.linear import Linear
from cs336_basics.transformer.rmsnorm import RMSNorm




class OutputLayer(nn.Module):
    def __init__(self, d_model: int, vocab_size: int, use_norm: bool = False):
        super().__init__()
        self.linear = Linear(d_model, vocab_size)
        self.norm = RMSNorm(d_model) if use_norm else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        logits = self.linear(x)
        return logits


class TransformerLM(nn.Module):
    def __init__(self,
        vocab_size: int,
        num_layers: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        use_rope: bool,
        rope_theta: float = 10000.0,
        max_seq_len: int = 2048): # max_seq_len = context_length

        super().__init__()
        
        self.token_embeddings = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([TransformerBlock(d_model, 
            num_heads, 
            d_ff, 
            use_rope=use_rope, 
            theta=rope_theta, 
            max_seq_len=max_seq_len) for _ in range(num_layers)])

        self.final_norm = RMSNorm(d_model)
        self.final_linear = Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Input:
            x: (batch_size, seq_len), a matrix of indices in the vocab_size
        Output:
            logits: (batch_size, seq_len, vocab_size)
        '''
        x = self.token_embeddings(x)
        for layer in self.layers:
            x = layer(x)
        x = self.final_norm(x)
        logits = self.final_linear(x)
        return logits
