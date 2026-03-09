import torch.nn as nn
import torch
from cs336_basics.transformer.linear import Linear

def SiLU(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)

class SwiGLUFFW(nn.Module):
    def __init__(self, d_model: int, d_ff: int | None = None):
        super().__init__()
        self.d_model = d_model
        if d_ff is None:
            self.d_ff = (int(8 / 3 * d_model) + 63) // 64 * 64
            
        else:
            self.d_ff = d_ff
        
        self.w1 = Linear(self.d_model, self.d_ff)
        self.w2 = Linear(self.d_ff, self.d_model)
        self.w3 = Linear(self.d_model, self.d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: ..., d_model
        x_1 = SiLU(self.w1(x)) # 
        x_2 = self.w3(x)
        return self.w2(x_1 * x_2)

