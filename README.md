# Paper Architecture Implementations

This repository contains clean, modular PyTorch implementations of architectures from various deep learning papers on Large Language Models. **No training code is included**—only the model architectures, attention mechanisms, and building blocks.

## Implemented Architectures

### 1. Llama 3 (`core/nn/llama3/`)
A full transformer implementation featuring:
- **Grouped Query Attention (GQA)** with standard KV caching for efficient inference
- **SwiGLU Feed-Forward Network** with hidden dimension calculation
- **RMSNorm** pre-normalization
- **Rotary Positional Embeddings (RoPE)**
- Causal masking and inference-mode token-by-token generation support

**Files:** `llama.py`, `transformer.py`, `grouped_query_attn.py`, `ffn.py`

---

### 2. DeepSeek-V2 (`core/nn/deepseekv2/`)
Two key components from the DeepSeek-V2 paper:

- **Multi-Head Latent Attention (MLA)**
  - Low-rank compression for queries (`W_dq`, `W_uq`) and key-values (`W_dkv`, `W_uk`, `W_uv`)
  - Decoupled RoPE for positional information (`W_qr`, `W_kr`)
  - KV latent caching for memory-efficient inference

- **Mixture-of-Experts (MoE)**
  - Shared experts + routed experts architecture
  - Group-limited routing with configurable `n_limited_groups`
  - Softmax or sigmoid scoring functions with learned expert bias
  - Top-k gating with route scaling

**Files:** `multihead_latent_attn.py`, `moe.py`

---

### 3. Linear Attention (`core/nn/linearAttention/`)
Kernelized attention mechanisms that reduce complexity from O(n²) to O(n):

- **`LinearAttention`** – Full (non-causal) linear attention with ELU/ReLU feature maps and KV-sum caching for O(1) memory inference
- **`CausalLinearSelfAttention`** – Causal variant using cumulative sums (`cumsum`) for autoregressive modeling

Both use explicit cache buffers (`CACHE_K_SUM`, `CACHE_KV_SUM`) for efficient incremental decoding.

**Files:** `linear_attn.py`

---

### 4. Hedgehog (`core/nn/hedgehog/`)
A learnable linear attention framework that trains feature maps to approximate standard softmax attention:

- **`HedgehogFeatureMap`** – FFN-based feature map initialized as identity, trained via soft-label cross-entropy against true softmax attention
- **`HedgehogAttention`** – Wrapper module that freezes a base attention layer and learns feature maps via forward hooks
- Supports arbitrary base attention layers through the `AttentionLayer` abstraction

**Files:** `hedgehog.py`, `attn.py`

---

### 5. m-HyperCell (mHC) (`core/nn/mHC/`)
Dynamic many-to-many channel mapping using doubly-stochastic matrices:

- **`MHCMapping`** – Generates dynamic routing weights (`H_pre`, `H_post`, `H_res`) via gating and linear projections
- **`MHCBlock`** – Wraps any `nn.Module` with mHC mappings using `torch.einsum`
- **`Sinkhorn-Knopp`** – Log-space and standard iterative algorithms for doubly-stochastic normalization

**Files:** `mhc_block.py`, `mhc_mapping.py`, `sk/sinkhorn_knopp.py`

---

## Repository Structure

```
core/
├── nn/
│   ├── llama3/              # Llama 3 transformer
│   ├── deepseekv2/          # DeepSeek-V2 MLA + MoE
│   ├── linearAttention/     # Linear / Causal Linear Attention
│   ├── hedgehog/            # Learnable linear attention
│   ├── mHC/                 # m-HyperCell dynamic routing
│   ├── lora/                # (Placeholder)
│   └── quantization/        # (Placeholder)
├── pos_embed/
│   └── rotary_embed.py      # RoPE + YaRN-scaled RoPE for MLA
├── norm/
│   └── rms_norm.py          # RMSNorm implementation
├── utils/
│   ├── params.py            # Hyperparameter dataclasses
│   └── device.py            # Auto device selection
└── einsum/                  # Einsum utilities
```

## Installation

```bash
pip install -r requirements.txt
```

Core dependency: **PyTorch 2.x**

## Quick Start

### Llama 3
```python
from core.nn.llama3.llama import LlamaTransformer
from core.utils.params import ParamsLlama3

params = ParamsLlama3(device='cuda')
model = LlamaTransformer(params)

# Inference
tokens = torch.randint(0, params.vocab_size, (2, 8))
output = model(tokens, start_pos=0)  # [B, T, vocab_size]
```

### DeepSeek-V2 MLA
```python
from core.nn.deepseekv2.multihead_latent_attn import MultiHeadLatentAttention
from core.utils.params import DeepSeekV2

params = DeepSeekV2(device='cuda')
mla = MultiHeadLatentAttention(params)

# First pass (prompt)
out = mla(x, start_pos=0, freqs_cis=freqs_cis, mask=mask)

# Subsequent tokens use KV cache automatically in eval mode
out = mla(next_token, start_pos=8, freqs_cis=next_freqs, mask=None)
```

### Linear Attention
```python
from core.nn.linearAttention.linear_attn import LinearAttention, elu_feature_map

attn = LinearAttention(elu_feature_map).eval()
out = attn(q, k, v)  # q/k/v shape: [B, T, H, D]
# KV sums are cached automatically when model.eval()
```

## Key Design Decisions

- **Modularity**: Each paper is self-contained; components can be mixed (e.g., `AttentionLayer` abstraction in Hedgehog)
- **Inference-First**: KV caches and state buffers are implemented wherever applicable (GQA, MLA, Linear Attention)
- **Einsum-Based**: Heavy use of `torch.einsum` for readable, shape-explicit tensor operations
- **Dataclass Configs**: All hyperparameters are centralized in `core/utils/params.py` via `@dataclass`

## References

| Module | Paper / Source |
|--------|---------------|
| Llama 3 | [Llama 3 Model Card](https://github.com/meta-llama/llama3) |
| DeepSeek-V2 | [DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model](https://arxiv.org/abs/2405.04434) |
| Linear Attention | [Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention](https://arxiv.org/abs/2006.16236) |
| Hedgehog | [The Hedgehog & the Porcupine: Two Linear Attention Models](https://arxiv.org/abs/2310.18680) |
| mHC | m-HyperCell dynamic channel routing |
