import torch
from torch import nn

from core.nn.mHC.mhc_mapping import MHCMapping


class MHCBlock(nn.Module):

    def __init__(self, hidden_dim: int, layer: nn.Module, expansion_rate: int = 4, sk_iter: int = 20):
        super().__init__()
        self.mhc_mapping = MHCMapping(hidden_dim=hidden_dim, expansion_rate=expansion_rate, sk_iter=sk_iter)
        self.layer = layer

    def forward(self, x: torch.Tensor, start_pos=None, mask=None, freqs_cis=None) -> torch.Tensor:
        H_pre, H_post, H_res = self.mhc_mapping(x)

        h = torch.einsum("btnc, btn -> btc", x, H_pre)
        h = self.layer(h, start_pos=start_pos, mask=mask, freqs_cis=freqs_cis)
        h = torch.einsum("btc, btn -> btnc", h, H_post)

        x = torch.einsum("btNc, btNn -> btnc", x, H_res)
        return x + h


if __name__ == '__main__':
    C = 64
    btnc = 2, 3, 4, C
    x = torch.randn(*btnc)
    layer = nn.Linear(C, C)

    m = MHCBlock(C, layer)
    y = m(x)
    assert y.shape == x.shape
