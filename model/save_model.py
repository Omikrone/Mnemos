
import numpy as np
from dataclasses import dataclass

@dataclass
class AttentionParams:
    query_matrix: np.ndarray
    key_matrix: np.ndarray
    value_matrix: np.ndarray

@dataclass
class MLPParams:
    w_up: np.ndarray
    b_up: np.ndarray
    w_down: np.ndarray
    b_down: np.ndarray

@dataclass
class LayerNormParams:
    gamma: np.ndarray
    beta: np.ndarray
    eps: float

@dataclass
class TransformerBlockParams:
    ln1: LayerNormParams
    w_out: np.ndarray
    b_out: np.ndarray
    self_attention: AttentionParams
    ln2: LayerNormParams
    mlp: MLPParams

@dataclass
class ModelParams:
    embedding_matrix: np.ndarray
    position_matrix: np.ndarray
    transformer_block_params: TransformerBlockParams