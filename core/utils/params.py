from dataclasses import dataclass, field, asdict
from typing import Optional

import torch


@dataclass
class ParamsLLama3:
    dim: int
    ffn_dim_multiplier: float
    multiple_of: int
    n_heads: int
    n_kv_heads: int
    n_layers: int
    norm_eps: float
    rope_theta: float
    use_scaled_rope: bool
    vocab_size: int
    max_batch_size: int
    max_seq_length: int
    n_kv_head_rep: int
    dim_head: int
    ffn_dim: Optional[int] = None
    device: Optional[torch.device] = None

    def __post_init__(self):
        if self.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"Warning: using default device ({self.device}) as device was not provided")

    def to_dict(self):
        return asdict(self)


def params_llama3(device: torch.device):
    return ParamsLLama3(
        dim=4096,
        ffn_dim_multiplier=1.3,
        multiple_of=1024,
        n_heads=32,
        n_kv_heads=8,
        n_layers=32,
        norm_eps=1e-05,
        rope_theta=500000.0,
        use_scaled_rope=True,
        vocab_size=128256,
        max_batch_size=4,
        max_seq_length=128,
        n_kv_head_rep=4,
        dim_head=128,
        device=device,
    )


if __name__ == '__main__':
    device = torch.device("cpu")
    p = params_llama3(device)
    print(p.device)
    print(p.ffn_dim)
    p.ffn_dim = 1
    print(p.ffn_dim)
    d = p.to_dict()
