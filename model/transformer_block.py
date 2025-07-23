import numpy as np

from model.gradient import Param
from model.layer_norm import LayerNorm
from model.mlp import MLP
from model.attention import MultiHeadAttention
from model.save_model import TransformerBlockParams
from config.params import EMBEDDING_DIM, HIDDEN_DIM


class TransformerBlock:
    """ Transformer block class that combines self-attention and feed-forward layers with layer normalization. """
    
    last_x : np.ndarray
    ln1 : LayerNorm
    w_out : Param
    b_out : Param
    self_attention : MultiHeadAttention
    ln2 : LayerNorm
    mlp : MLP


    def __init__(self, vocab_size : int):
        """ Initialize the Transformer block with layer normalization, attention, and MLP layers. """

        self.ln1 = LayerNorm(EMBEDDING_DIM)
        self.w_out = Param(np.random.randn(EMBEDDING_DIM, vocab_size) * 0.02)
        self.b_out = Param(np.zeros((vocab_size,)))
        self.self_attention = MultiHeadAttention()
        self.ln2 = LayerNorm(EMBEDDING_DIM)
        self.mlp = MLP()


    @classmethod
    def from_params(cls, params: TransformerBlockParams) -> 'TransformerBlock':
        """ Create a TransformerBlock instance from saved parameters. """

        instance = cls(vocab_size=params.w_out.shape[1])
        instance.ln1 = LayerNorm.from_params(params.ln1)
        instance.w_out = Param(params.w_out)
        instance.b_out = Param(params.b_out)
        instance.self_attention = MultiHeadAttention.from_params(params.self_attention)
        instance.ln2 = LayerNorm.from_params(params.ln2)
        instance.mlp = MLP.from_params(params.mlp)
        return instance


    def forward(self, inputs : np.ndarray):
        """ Forward pass through the Transformer block. """

        # Layer normalization and residual connection for the self-attention sub-layer
        norm_x = self.ln1.forward(inputs)
        attn_out = self.self_attention.forward(norm_x)
        x = inputs + attn_out

        # Layer normalization and residual connection for the MLP sub-layer
        norm_x = self.ln2.forward(x)
        mlp_out = self.mlp.feed_forward(norm_x)
        x2 = x + mlp_out
        self.last_x = x2

        # Final linear transformation to produce logits
        logits = x2 @ self.w_out.value + self.b_out.value

        return logits


    def backward(self, grad_logits) -> np.ndarray:
        """ Backward pass through the Transformer block. """

        # Compute gradients for the output layer
        self.w_out.gradient = np.einsum('btd,btv->dv', self.last_x, grad_logits)
        self.b_out.gradient = np.sum(grad_logits, axis=(0, 1))

        grad_x2 = grad_logits @ self.w_out.value.T

        # Backpropagation through the MLP layer
        grad_mlp = self.mlp.backward(grad_x2)
        grad_ln2 = self.ln2.backward(grad_mlp)
        grad_x = grad_ln2 + grad_x2

        # Backpropagation through the self-attention layer
        grad_a = self.self_attention.backward(grad_x)
        grad_ln1 = self.ln1.backward(grad_a)
        grad_input = grad_ln1 + grad_x

        return grad_input
    

    def step(self, lr : float):
        """ Update the parameters of the Transformer block using gradient descent. """

        self.w_out.step(lr)
        self.b_out.step(lr)
        self.self_attention.step(lr)
        self.mlp.step(lr)
        self.ln1.step(lr)
        self.ln2.step(lr)


    def zero_grad(self):
        """ Reset the gradients of the Transformer block parameters. """

        self.w_out.zero_grad()
        self.b_out.zero_grad()
        self.self_attention.zero_grad()
        self.mlp.zero_grad()
        self.ln1.zero_grad()
        self.ln2.zero_grad()


    def get_parameters(self) -> TransformerBlockParams:
        """ Return the parameters of the Transformer block for saving. """

        return TransformerBlockParams(
            ln1=self.ln1.get_params(),
            w_out=self.w_out.value,
            b_out=self.b_out.value,
            self_attention=self.self_attention.get_params(),
            ln2=self.ln2.get_params(),
            mlp=self.mlp.get_parameters()
        )