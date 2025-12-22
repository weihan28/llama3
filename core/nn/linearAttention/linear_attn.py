import torch
from torch import nn
from torch.nn import functional as F


def elu_feature_map(x: torch.Tensor) -> torch.Tensor:
    return F.elu(x) + 1


class LinearAttention(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.dim = args.dim
        self.n_heads = args.attn.n_heads
        self.qk_dim = args.attn.qk_dim
        self.v_dim = args.attn.v_dim

        self.WQ = nn.Linear(self.dim, self.qk_dim, bias=False)
        self.WK = nn.Linear(self.dim, self.qk_dim, bias=False)
        self.WV = nn.Linear(self.dim, self.v_dim, bias=False)

        self.proj = nn.Linear(self.v_dim, self.dim, bias=False)

    def forward(self, x) -> torch.Tensor:
        # Q @ K^T @ V = [.., T, D] @ [.., D, T] @ [.., T, M]

        q = self.WQ(x) # [B, T, H, D]
        k = self.WK(x) # [B, T, H, D]
        v = self.WV(x) # [B, T, H, M]

        kv = torch.einsum("bthd, bthm -> bhdm", k, v) # K^T @ V
        z = 1/torch.einsum("bthd, bhd -> bth", q, k.sum(1)) # Σ_d q_t @ (K1+K2..KT)^T = Σ_d (q_t @ Σ_t (K^T))
        o = torch.einsum("bthd, bhdm, bth-> bthm", q, kv, z) # Σ_d (Q_bthd KV_bhdm Z_bth)
        return self.proj(o) # [B, T, H, e]

if __name__ == "__main__":
    from types import SimpleNamespace

    args = SimpleNamespace(
        dim=512,
        attn=SimpleNamespace(
            n_heads=8,
            qk_dim=64,
            v_dim=64
        )
    )

    attn = LinearAttention(args)
    b, t, h= 2,3,4
    x = torch.randn(b, t, h, args.dim)
    out = attn(x)
    print(out.shape)



