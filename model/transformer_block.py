import numpy as np
from model.mlp import MLP
from model.attention import Attention
from model.embeddings import EMBEDDING_DIMENSION


class TransformerBlock:
    
    w_out : np.ndarray
    b_out : np.ndarray
    self_attention : Attention
    mlp : MLP

    def __init__(self, vocab_size : int):
        self.w_out = np.random.randn(EMBEDDING_DIMENSION, 85) * 0.02
        self.b_out = np.zeros((vocab_size,))
        self.self_attention = Attention()
        self.mlp = MLP()

    def forward(self, inputs : np.ndarray):
        attn_out = self.self_attention.add_attention(inputs)
        x = inputs + attn_out

        mlp_out = self.mlp.feed_forward(inputs)
        x += mlp_out

        logits = x @ self.w_out + self.b_out

        return logits
