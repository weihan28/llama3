from torch import nn
from torch.nn import functional as F


# https://github.com/meta-llama/llama3/blob/main/llama/model.py#L202
def _get_ffn_dim(hidden_dim, ffn_dim_multiplier, multiple_of):
    hidden_dim = int(2 * hidden_dim / 3)
    # custom dim factor multiplier
    if ffn_dim_multiplier is not None:
        hidden_dim = int(ffn_dim_multiplier * hidden_dim)
    hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
    return hidden_dim


class FeedForwardSwiGLU(nn.Module):
    def __init__(self, params):
        super(FeedForwardSwiGLU, self).__init__()
        if params.ffn_dim is None:
            params.ffn_dim = _get_ffn_dim(4 * params.dim, params.ffn_dim_multiplier, params.multiple_of)
        self.w1 = nn.Linear(params.dim, params.ffn_dim, bias=False)
        self.w2 = nn.Linear(params.dim, params.ffn_dim, bias=False)
        self.proj = nn.Linear(params.ffn_dim, params.dim, bias=False)

    def forward(self, x):
        return self.proj(F.silu(self.w1(x)) * self.w2(x))


if __name__ == "__main__":
    import torch
    from core.utils.params import params_llama3

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    params = params_llama3(device)
    dummy_input = torch.randn(2, 8, params.dim)
    model = FeedForwardSwiGLU(params)
    print(dummy_input.shape)
    print(model(dummy_input).shape)
