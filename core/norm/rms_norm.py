import torch
from torch import nn


class RMSNorm(torch.nn.Module):
    def __init__(self, dim, norm_eps):
        super(RMSNorm, self).__init__()
        self.norm_eps = norm_eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.norm_eps)

    def forward(self, x):
        out = self._norm(x.float()).type_as(x)
        return out * self.weight  # (2, 8, DIM)


if __name__ == '__main__':
    dummy_inp = torch.randn(2, 8, 4096)
    norm = RMSNorm(4096, 1e-5)
    output = norm(dummy_inp)
    print(output.shape)
