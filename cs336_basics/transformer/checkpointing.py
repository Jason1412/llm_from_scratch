import torch
import typing
import os

def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]):
    """
    Given a model, optimizer, and an iteration number, serialize them to disk.
    """
    model_weights = model.state_dict()
    optimizer_weights = optimizer.state_dict()
    checkpoint = {
        'model_weights': model_weights,
        'optimizer_weights': optimizer_weights,
        'iteration': iteration
    }

    torch.save(checkpoint, out)


def load_checkpoint(
    src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    ):

    checkpoint = torch.load(src)
    model.load_state_dict(checkpoint['model_weights'])
    optimizer.load_state_dict(checkpoint['optimizer_weights'])
    return checkpoint['iteration']
