import torch


def resolve_device(configs):
    requested = str(getattr(configs, "device", "auto") or "auto").lower()

    if getattr(configs, "use_multi_gpu", False):
        if not torch.cuda.is_available():
            raise RuntimeError("--use_multi_gpu requires CUDA, but CUDA is not available.")
        local_rank = int(getattr(configs, "local_rank", 0))
        return torch.device(f"cuda:{local_rank}")

    if requested == "auto":
        gpu = int(getattr(configs, "gpu", 0))
        if torch.cuda.is_available():
            return torch.device(f"cuda:{gpu}")
        return torch.device("cpu")

    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"Requested device {requested}, but CUDA is not available.")
    return device


def model_dtype(configs, device):
    return torch.float16 if bool(getattr(configs, "use_amp", False)) and device.type == "cuda" else torch.float32
