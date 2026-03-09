import torch
import torch.nn as nn
from einops import einsum

class Linear(nn.Module):

    def __init__(self, 
        in_features: int, 
        out_features: int, 
        device: torch.device | None = None, 
        dtype: torch.dtype | None = None):

        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.dtype = dtype
        self.weight = nn.Parameter(torch.empty((out_features, in_features), device=device, dtype=dtype))
        sigma = (2./(in_features+out_features))**0.5
        nn.init.trunc_normal_(self.weight, std=sigma, a=-3.*sigma, b=3.*sigma)
        
    def forward(self, X: torch.Tensor) -> torch.Tensor:
    # X shape: ..., in_features
        return einsum(X, self.weight, "... in_features, out_features in_features -> ... out_features")


