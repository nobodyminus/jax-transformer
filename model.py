from typing import Optional, Mapping, Tuple

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np


class SelfAttention(hk.MultiHeadAttention):
    def __call__(self, q: jnp.ndarray, k: Optional[jnp.ndarray] = None,
                 v: Optional[jnp.ndarray] = None, mask: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        if k is None:
            k = q
        if v is None:
            v = q
        seq_len = q.shape[1]
        causal_mask = np.tril(np.ones((seq_len, seq_len)))
        if mask is not None:
            mask = mask * causal_mask
        else:
            mask = causal_mask
        return super.__call__(q, k, v, mask)


class DenseBlock(hk.Module):
    def __init__(self, init_scale: float, widening_factor: int = 4, name: Optional[str] = None) -> None:
        super().__init__(name=name)
        self._init_scale = init_scale
        self._widening_factor = widening_factor

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        hidden_size = x.shape[-1]
        initializer = hk.initializers.VarianceScaling(self._init_scale)
        x = hk.Linear(self._widening_factor, w_init=initializer)(x)
        x = jax.nn.gelu(x)
        return hk.Linear(hidden_size, w_init=initializer)(x)


class Transformer(hk.Module):
    def __init__(self, num_heads: int, num_layers: int, dropout: float, name: Optional[str] = None) -> None:
        super().__init__(name=name)
        self._num_layers = num_layers
        self._num_heads = num_heads
        self._dropout = dropout

    def __call__(self, h: jnp.ndarray, mask: Optional[jnp.ndarray], is_training: bool) -> jnp.ndarray:
        init_scale = 2. / self._num_layers
        if is_training:
            dropout = 0
        else:
            dropout = self._dropout
        if mask is not None:
            mask = mask[:, None, None, :]
        for i in range(self._num_layers):
            h_norm = layer_norm(h, name=f'h{i}_ln_1')
            h_attn = SelfAttention(num_heads=self._num_heads,
                                   key_size=64,
                                   w_init_scale=init_scale,
                                   name=f'h{i}_attn')(h_norm, mask=mask)
            h_attn = hk.dropout(hk.next_rng_key(), dropout, h_attn)
            h += h_attn
            h_norm = layer_norm(h, name=f'h{i}_ln_2')
            h_dense = DenseBlock(init_scale, name=f'h{i}_mlp')(h_norm)
            h_dense = hk.dropout(hk.next_rng_key(), dropout, h_dense)
            h += h_dense
        h = layer_norm(h, name='ln_f')
        return h


def layer_norm(x: jnp.ndarray, name: Optional[str] = None) -> jnp.ndarray:
    return hk.LayerNorm(axis=-1, create_scale=True, create_offset=True, name=name)(x)


def embeddings(data: Mapping[str, jnp.ndarray], vocab_size: int, d_model: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
    tokens = data['obs']
    input_mask = jnp.greater(tokens, 0)
    seq_len = tokens.shape[1]
    embed_init = hk.initializers.TruncatedNormal(stddev=0.02)
    token_embedding_map = hk.Embed(vocab_size, d_model, w_init=embed_init)
    token_embs = token_embedding_map(tokens)
    positional_embeddings = hk.get_parameter('pos_embs', [seq_len, d_model], init=embed_init)
    input_embeddings = token_embs + positional_embeddings
    return input_embeddings, input_mask
