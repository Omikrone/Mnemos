import numpy as np

from model.gradient import Param
from model.layer_norm import LayerNorm
from model.linear import Linear
from model.mlp import MLP
from model.attention import Attention
from model.embeddings import EMBEDDING_DIMENSION
from model.save_model import TransformerBlockParams


class TransformerBlock:
    
    last_x : np.ndarray
    ln1 : LayerNorm
    w_out : Param
    b_out : Param
    self_attention : Attention
    ln2 : LayerNorm
    mlp : MLP

    def __init__(self, vocab_size : int):
        self.ln1 = LayerNorm(EMBEDDING_DIMENSION)
        self.w_out = Param(np.random.randn(EMBEDDING_DIMENSION, vocab_size) * 0.02)
        self.b_out = Param(np.zeros((vocab_size,)))
        self.self_attention = Attention()
        self.ln2 = LayerNorm(EMBEDDING_DIMENSION)
        self.mlp = MLP()

    @classmethod
    def from_params(cls, params: TransformerBlockParams) -> 'TransformerBlock':
        instance = cls(vocab_size=params.w_out.shape[1])
        instance.ln1 = LayerNorm.from_params(params.ln1)
        instance.w_out = Param(params.w_out)
        instance.b_out = Param(params.b_out)
        instance.self_attention = Attention.from_params(params.self_attention)
        instance.ln2 = LayerNorm.from_params(params.ln2)
        instance.mlp = MLP.from_params(params.mlp)
        return instance

    def forward(self, inputs : np.ndarray):
        norm_x = self.ln1.forward(inputs)
        attn_out = self.self_attention.add_attention(norm_x)
        x = inputs + attn_out

        norm_x = self.ln2.forward(x)
        mlp_out = self.mlp.feed_forward(norm_x)
        x2 = x + mlp_out
        self.last_x = x2

        logits = x2 @ self.w_out.value + self.b_out.value

        return logits

    def backward(self, grad_logits) -> np.ndarray:
        self.w_out.gradient = np.einsum('btd,btv->dv', self.last_x, grad_logits)
        self.b_out.gradient = np.sum(grad_logits, axis=(0, 1))

        grad_x2 = grad_logits @ self.w_out.value.T

        grad_mlp = self.mlp.backward(grad_x2)
        grad_ln2 = self.ln2.backward(grad_mlp)
        grad_x = grad_ln2 + grad_x2

        grad_a = self.self_attention.backward(grad_x)
        grad_ln1 = self.ln1.backward(grad_a)
        grad_input = grad_ln1 + grad_x

        return grad_input
    

    def step(self, lr : float):
        self.w_out.step(lr)
        self.b_out.step(lr)
        self.self_attention.step(lr)
        self.mlp.step(lr)
        self.ln1.step(lr)
        self.ln2.step(lr)

    def zero_grad(self):
        self.w_out.zero_grad()
        self.b_out.zero_grad()
        self.self_attention.zero_grad()
        self.mlp.zero_grad()
        self.ln1.zero_grad()
        self.ln2.zero_grad()

    def get_parameters(self):
        return TransformerBlockParams(
            ln1=self.ln1.get_params(),
            w_out=self.w_out.value,
            b_out=self.b_out.value,
            self_attention=self.self_attention.get_parameters(),
            ln2=self.ln2.get_params(),
            mlp=self.mlp.get_parameters()
        )