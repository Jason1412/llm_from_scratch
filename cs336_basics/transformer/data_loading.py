from numpy.typing import NDArray
import numpy as np
import torch

def data_loading(dataset: NDArray,
                batch_size: int,
                context_length: int,
                device: str) -> tuple[torch.Tensor, torch.Tensor]:
    '''
    Input:
        dataset: 1D array
    Output:
        inputs: (batch_size, seq_length) 
        targets: (batch_size, seq_length)
    '''
    # Indices must be such that we can take context_length + 1 tokens
    max_idx = len(dataset) - context_length - 1
    
    # Sample random starting indices
    idxs = torch.randint(0, max_idx + 1, (batch_size,))
    
    batch_data = [dataset[i : i + context_length + 1] for i in idxs]
    batch_tensor = torch.tensor(np.array(batch_data), dtype=torch.long).to(device)
    
    inputs = batch_tensor[:, :-1]
    targets = batch_tensor[:, 1:]
    
    return inputs, targets