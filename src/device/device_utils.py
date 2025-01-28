import torch

def get_torch_device():
    """
    Determines the best available torch device (CPU, CUDA, or MPS).
    """
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")
