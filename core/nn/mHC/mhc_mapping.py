import torch
from torch import nn

from core.nn.mHC.sk.sinkhorn_knopp import sinkhorn_knopp_log


class MHCMapping(nn.Module):

    def __init__(self, hidden_dim, expansion_rate=4, sk_iter=20):
        super().__init__()
        self.n = expansion_rate
        self.C = hidden_dim
        self.t_max = sk_iter

        # scalar gating factors (init to 0.01)
        self.a_pre = nn.Parameter(torch.ones(()) * 1e-2)
        self.a_post = nn.Parameter(torch.ones(()) * 1e-2)
        self.a_res = nn.Parameter(torch.ones(()) * 1e-2)

        # bias vectors
        self.b_pre = nn.Parameter(torch.zeros(self.n))
        self.b_post = nn.Parameter(torch.zeros(self.n))
        self.b_res = nn.Parameter(torch.zeros(self.n * self.n))

        # projection matrices
        self.W_pre = nn.Linear(self.n * self.C, self.n, bias=False)
        self.W_post = nn.Linear(self.n * self.C, self.n, bias=False)
        self.W_res = nn.Linear(self.n * self.C, self.n * self.n, bias=False)

    def forward(self, x):
        """ Dynamically generate mapping weights.


        Output shape:\n
        - H_pre of shape [b,t,n]\n
        - H_post of shape [b,t,n]\n
        - W_pre of shape [b,t,n,n]

        :param x: input of shape [B, T, N, C], where N is the expansion rate.
        :return: Tuple(H_pre, H_post, H_res)
        """
        # x is of shape [B, T, N, C]
        x = torch.flatten(x, start_dim=-2, end_dim=-1)  # [b,t,n*C]

        # generate dynamic weights
        H_pre = self.a_pre * self.W_pre(x) + self.b_pre  # [b,t,n]
        H_post = self.a_post * self.W_post(x) + self.b_post  # [b,t,n]
        H_res = self.a_res * self.W_res(x) + self.b_res  # [b,t,n^2]

        # activation and make doubly stochastic
        H_pre = torch.sigmoid(H_pre)
        H_post = 2 * torch.sigmoid(H_post)
        H_res = sinkhorn_knopp_log(H_res.view(*x.shape[:-1], self.n, self.n), n_iters=self.t_max)
        return H_pre, H_post, H_res


if __name__ == '__main__':
    B, T, N, C = 2, 3, 4, 64
    x = torch.randn(B, T, N, C)

    m = MHCMapping(hidden_dim=C, expansion_rate=4, sk_iter=20)
    H_pre, H_post, H_res = m(x)
    print(H_pre.shape)
    print(H_post.shape)
    print(H_res.shape)
