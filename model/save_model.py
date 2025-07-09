
import numpy as np
from dataclasses import dataclass

@dataclass
class AttentionParams:
    query_matrix: np.ndarray
    key_matrix: np.ndarray
    value_matrix: np.ndarray

@dataclass
class MLPParams:
    w1: np.ndarray
    b1: np.ndarray
    w2: np.ndarray
    b2: np.ndarray

@dataclass
class TransformerBlockParams:
    w_out: np.ndarray
    b_out: np.ndarray
    self_attention: AttentionParams
    mlp: MLPParams

@dataclass
class ModelParams:
    embedding_matrix: np.ndarray
    position_matrix: np.ndarray
    transformer_block_params: TransformerBlockParams