import numpy as np
from dataclasses import dataclass


@dataclass
class AttentionParams:
    """ Parameters for the attention mechanism. """

    query_matrix: np.ndarray
    key_matrix: np.ndarray
    value_matrix: np.ndarray


@dataclass
class MultiHeadAttentionParams:
    """ Parameters for teh multi head attention mechanism """

    w_out : np.ndarray
    b_out : np.ndarray
    attention_heads : list[AttentionParams]


@dataclass
class MLPParams:
    """ Parameters for the Multi-Layer Perceptron (MLP). """

    w_up: np.ndarray
    b_up: np.ndarray
    w_down: np.ndarray
    b_down: np.ndarray


@dataclass
class LayerNormParams:
    """ Parameters for Layer Normalization. """

    gamma: np.ndarray
    beta: np.ndarray
    eps: float


@dataclass
class TransformerBlockParams:
    """ Parameters for a Transformer block. """

    ln1: LayerNormParams
    w_out: np.ndarray
    b_out: np.ndarray
    self_attention: AttentionParams
    ln2: LayerNormParams
    mlp: MLPParams


@dataclass
class ModelParams:
    """ Parameters for the entire model. """

    embedding_matrix: np.ndarray
    position_matrix: np.ndarray
    transformer_block_params: TransformerBlockParams