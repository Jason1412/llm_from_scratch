import torch 
import torch.nn as nn
from einops import einsum, reduce

class RMSNorm(nn.Module):
    def __init__(self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None):

        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.device = device
        self.dtype = dtype

        self.weight = nn.Parameter(torch.empty(d_model, device=device, dtype=dtype))
        nn.init.ones_(self.weight)


        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Args:
            X: (... d_model), e.g. (batch_size, sequence_length, d_model)
        Returns:
            Output: (... d_model)
        '''
        
        in_dtype = x.dtype

        x = x.to(torch.float32)

        rms = torch.sqrt(reduce(x**2, "... d_model -> ... 1", "mean") + self.eps)
        # rms.shape = (batch_size, sequence_length, 1)

        rms_norm = (x / rms) * self.weight

        return rms_norm.to(in_dtype)
