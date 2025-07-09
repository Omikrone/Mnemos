import numpy as np

from model.gradient import Param
from model.linear import Linear
from model.mlp import MLP
from model.attention import Attention
from model.embeddings import EMBEDDING_DIMENSION
from model.save_model import TransformerBlockParams


class TransformerBlock:
    
    last_x : np.ndarray
    w_out : Param
    b_out : Param
    self_attention : Attention
    mlp : MLP

    def __init__(self, vocab_size : int):
        self.w_out = Param(np.random.randn(EMBEDDING_DIMENSION, vocab_size) * 0.02)
        self.b_out = Param(np.zeros((vocab_size,)))
        self.self_attention = Attention()
        self.mlp = MLP()

    @classmethod
    def from_params(cls, params: TransformerBlockParams) -> 'TransformerBlock':
        instance = cls(vocab_size=params.w_out.shape[1])
        instance.w_out = Param(params.w_out)
        instance.b_out = Param(params.b_out)
        instance.self_attention = Attention.from_params(params.self_attention)
        instance.mlp = MLP.from_params(params.mlp)
        return instance

    def forward(self, inputs : np.ndarray):
        attn_out = self.self_attention.add_attention(inputs)
        x = inputs + attn_out

        mlp_out = self.mlp.feed_forward(inputs)
        x += mlp_out
        self.last_x = x

        logits = x @ self.w_out.value + self.b_out.value

        return logits

    def backward(self, grad_logits) -> np.ndarray:
        self.w_out.gradient = np.einsum('btd,btv->dv', self.last_x, grad_logits)
        self.b_out.gradient = np.sum(grad_logits, axis=(0, 1))

        grad_x = grad_logits @ self.w_out.value.T
        grad_x = self.mlp.backward(grad_x)
        grad_a = self.self_attention.backward(grad_x)
        return grad_a
    
    def step(self, lr : float):
        self.w_out.step(lr)
        self.b_out.step(lr)
        self.self_attention.step(lr)
        self.mlp.step(lr)

    def zero_grad(self):
        self.w_out.zero_grad()
        self.b_out.zero_grad()
        self.self_attention.zero_grad()
        self.mlp.zero_grad()

    def get_parameters(self):
        return TransformerBlockParams(
            w_out=self.w_out.value,
            b_out=self.b_out.value,
            self_attention=self.self_attention.get_parameters(),
            mlp=self.mlp.get_parameters()
        )