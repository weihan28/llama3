import torch


def get_device(device: str = None):
    if device is not None:
        return torch.device(device)

    # auto detect
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    return device


if __name__ == "__main__":
    print(get_device())
