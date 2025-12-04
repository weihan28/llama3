import torch


def precompute_freqs_cis(dim, end, theta=10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis, x):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1]), f"freqs_cis.shape: {freqs_cis.shape}, x.shape: {x.shape}"
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(xq, xk, freqs_cis):
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


if __name__ == '__main__':
    DIM = 4096  # Llama3 Table 3.
    N_HEADS = 32  # Llama3 Table 3.
    N_KV_HEADS = 8  # With 8 key-value heads to improve inference speed and to reduce the size (llama3)
    ROPE_THETA = 500000  # We increase the RoPE base frequency hyperparameter to 500,000 (llama3)
    MAX_SEQ_LEN = 128  # Just optional depending on your specs.
    HEAD_DIM = DIM // N_HEADS  # Divide dimension by number of heads to get dimension per head.

    dummy_inp1 = torch.randn(2, 8, N_HEADS, HEAD_DIM)
    dummy_inp2 = torch.randn(2, 8, N_KV_HEADS, HEAD_DIM)

    dummy_freqs_cis = precompute_freqs_cis(HEAD_DIM, MAX_SEQ_LEN * 2, ROPE_THETA)
    dummy_freqs_cis = dummy_freqs_cis[0: 0 + 8]

    out1, out2 = apply_rotary_emb(dummy_inp1, dummy_inp2, dummy_freqs_cis)
    print("-" * 30)
    print(out1.shape)
    print(out2.shape)
