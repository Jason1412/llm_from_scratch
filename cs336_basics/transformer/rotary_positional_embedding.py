import torch
import torch.nn as nn
from einops import rearrange


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device


        # 1. Compute the inverse frequencies (theta_i)
        # Shape: (d_k // 2)
        powers = torch.arange(0, d_k, 2, device=device).float()
        theta_freqs = 1.0 / (theta ** (powers / d_k))

        # 2. Compute the position indices (m)
        # Shape: (max_seq_len)
        m = torch.arange(max_seq_len, device=device)

        # 3. Create the frequency matrix
        # Shape: (max_seq_len, d_k // 2)
        freqs = torch.outer(m, theta_freqs)

        # 4. Convert to polar complex form: e^(i * m * theta)
        # Shape: (max_seq_len, d_k // 2)
        freqs_complex = torch.polar(torch.ones_like(freqs), freqs)

        self.register_buffer('freqs_complex', freqs_complex)



    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        '''
        Args:
            x: shape (..., seq_len, d_k)
            token_positions: shape (..., seq_len)

        Returns:
            torch.Tensor: shape (..., seq_len, d_k)
        
        '''

        # 1. Reshape x into complex numbers
        # (..., seq_len, d_k) -> (..., seq_len, d_k//2) complex

        x_complex = rearrange(x.float(), '... s (d c) -> ... s d c', c=2)
        # The last dimension of x_complex will be split into real and imaginary parts  
        # After view_as_complex(), the last dimension will be gone
        x_complex = torch.view_as_complex(x_complex)

        # 2. Index into precomputed frequencies
        # token_positions shape: (*, max_seq_len)
        # freqs_complex shape: (max_seq_len, d_k // 2)
        # batch_freqs shape: (*, max_seq_len, d_k // 2)
        batch_freqs = self.freqs_complex[token_positions]

        # 3. Apply rotation via complex multiplication
        # Shape: = (..., seq_len, d_k//2) * (*, max_seq_len, d_k // 2)
        x_rotated = x_complex * batch_freqs

        # Extract the real and imaginary parts of the complex number directly to create 2 dimensions as the last dimension
        x_out = torch.view_as_real(x_rotated)
        x_out = rearrange(x_out, '... s d c -> ... s (d c)')

        return x_out.type_as(x)


