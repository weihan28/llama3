import torch
from torch import nn

from core.nn.transformer import TransformerBlock
from core.norm.rms_norm import RMSNorm
from core.pos_embed.rotary_embed import precompute_freqs_cis
from core.utils.params import ParamsLLama3


class LlamaTransformer(nn.Module):
    def __init__(self, params: ParamsLLama3):
        super(LlamaTransformer, self).__init__()
        self.token_embeddings = nn.Embedding(params.vocab_size, params.dim)

        self.layers = nn.ModuleList()
        for _ in range(params.n_layers):
            self.layers.append(TransformerBlock(params))
        self.norm = RMSNorm(params.dim, params.norm_eps)
        self.output = nn.Linear(params.dim, params.vocab_size, bias=False)

        self.freqs_cis = precompute_freqs_cis(
            params.dim_head,
            2 * params.max_seq_length,
            params.rope_theta
        )

    @torch.inference_mode()
    def forward(self, tokens, start_pos):
        B, T = tokens.shape
        h = self.token_embeddings(tokens)  # (B, T, dim)
        self.freqs_cis = self.freqs_cis.to(tokens.device)
        freqs_cis = self.freqs_cis[start_pos:start_pos + T]

        mask = None
        if T > 1:  # because of KV cache, we process only 1 token except for the first run, which has a seq_len>1.
            mask = torch.full((T, T), float('-inf'), device=tokens.device)
            mask = torch.triu(mask, diagonal=1).to(tokens.device)  # only get upper triangle, lower triangle all 0.

        for layer in self.layers:
            h = layer(h, start_pos, freqs_cis, mask)  # (B, T, dim)

        return self.output(self.norm(h))  # (B, T, vocab_size)


if __name__ == "__main__":
    from core.utils.params import params_llama3
    from core.utils.device import get_device

    device = get_device('cpu')
    params = params_llama3(device)

    dummy_tokens = torch.rand(2, 8, device=params.device).long()  # Use rand instead of randn, as nn.Embeddings dont accept neg nums.
    dummy_start_pos = 0

    llama3 = LlamaTransformer(params).to(params.device)
    output = llama3(dummy_tokens, dummy_start_pos)

    print(dummy_tokens.shape)
    print(output.shape)  # (B, seq, vocab_size)
