"""Decoder-only TinyGPT with explicitly implemented causal attention."""

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from config import Config


class CausalSelfAttention(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model)
        self.projection = nn.Linear(config.d_model, config.d_model)
        self.attention_dropout = nn.Dropout(config.dropout)
        self.output_dropout = nn.Dropout(config.dropout)
        self.register_buffer(
            "causal_mask", torch.ones(config.context_length, config.context_length, dtype=torch.bool).tril(),
            persistent=False,
        )

    def forward(self, hidden: Tensor, input_mask: Tensor | None = None) -> Tensor:
        batch, length, width = hidden.shape
        q, k, v = (
            tensor.reshape(batch, length, self.n_heads, self.head_dim).transpose(1, 2)
            for tensor in self.qkv(hidden).chunk(3, dim=-1)
        )
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        allowed = self.causal_mask[:length, :length][None, None, :, :]
        if input_mask is not None:
            allowed = allowed & input_mask[:, None, None, :]
        # A completely masked query produces zero attention rather than NaNs.
        has_key = allowed.any(dim=-1, keepdim=True)
        scores = scores.masked_fill(~allowed, float("-inf"))
        scores = torch.where(has_key, scores, 0.0)
        probabilities = F.softmax(scores.float(), dim=-1).masked_fill(~allowed, 0.0).to(v.dtype)
        attended = self.attention_dropout(probabilities) @ v
        attended = attended.transpose(1, 2).contiguous().reshape(batch, length, width)
        output = self.output_dropout(self.projection(attended))
        if input_mask is not None:
            output = output.masked_fill(~input_mask[:, :, None], 0.0)
        return output


class MLP(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.expand = nn.Linear(config.d_model, config.mlp_ratio * config.d_model)
        self.activation = nn.GELU()
        self.projection = nn.Linear(config.mlp_ratio * config.d_model, config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, hidden: Tensor) -> Tensor:
        return self.dropout(self.projection(self.activation(self.expand(hidden))))


class Block(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.attention_norm = nn.LayerNorm(config.d_model)
        self.attention = CausalSelfAttention(config)
        self.mlp_norm = nn.LayerNorm(config.d_model)
        self.mlp = MLP(config)

    def forward(self, hidden: Tensor, input_mask: Tensor | None = None) -> Tensor:
        hidden = hidden + self.attention(self.attention_norm(hidden), input_mask)
        return hidden + self.mlp(self.mlp_norm(hidden))


class TinyGPT(nn.Module):
    def __init__(self, vocab_size: int, config: Config | None = None):
        super().__init__()
        if vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        self.config = config or Config()
        self.vocab_size = vocab_size
        c = self.config
        self.token_embedding = nn.Embedding(vocab_size, c.d_model)
        self.position_embedding = nn.Embedding(c.context_length, c.d_model)
        self.embedding_dropout = nn.Dropout(c.dropout)
        self.blocks = nn.ModuleList(Block(c) for _ in range(c.n_layers))
        self.final_norm = nn.LayerNorm(c.d_model)
        self.apply(self._initialize)
        for block in self.blocks:
            for projection in (block.attention.projection, block.mlp.projection):
                nn.init.normal_(projection.weight, std=c.init_std / math.sqrt(2 * c.n_layers))
        self.lm_head = nn.Linear(c.d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight

    def _initialize(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=self.config.init_std)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, inputs: Tensor, input_mask: Tensor | None = None) -> Tensor:
        if inputs.ndim != 2 or not 0 < inputs.shape[1] <= self.config.context_length:
            raise ValueError("Expected [B, T] input with 1 <= T <= context_length")
        if input_mask is not None and (input_mask.shape != inputs.shape or input_mask.dtype != torch.bool):
            raise ValueError("input_mask must be boolean and match the input shape")
        positions = torch.arange(inputs.shape[1], device=inputs.device)
        hidden = self.embedding_dropout(self.token_embedding(inputs) + self.position_embedding(positions))
        for block in self.blocks:
            hidden = block(hidden, input_mask)
        # Raw logits: cross_entropy applies log-softmax; generation applies softmax.
        return self.lm_head(self.final_norm(hidden))
