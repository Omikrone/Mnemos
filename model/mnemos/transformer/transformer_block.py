from mnemos import xp
from mnemos.transformer.dropout import Dropout
from mnemos.transformer.layer_norm import LayerNorm
from mnemos.transformer.mlp import MLP
from mnemos.transformer.attention import MultiHeadAttention
from mnemos.transformer.save_model import TransformerBlockParams
from mnemos.config.params import DROPOUT_RATE, EMBEDDING_DIM


class TransformerBlock:
    """ Transformer block class that combines self-attention and feed-forward layers with layer normalization. """
    
    ln1 : LayerNorm
    self_attention : MultiHeadAttention
    ln2 : LayerNorm
    mlp : MLP
    dropout : Dropout


    def __init__(self):
        """ Initialize the Transformer block with layer normalization, attention, and MLP layers. """

        self.ln1 = LayerNorm(EMBEDDING_DIM)
        self.self_attention = MultiHeadAttention()
        self.ln2 = LayerNorm(EMBEDDING_DIM)
        self.mlp = MLP()
        self.dropout = Dropout(DROPOUT_RATE)


    @classmethod
    def from_params(cls, params: TransformerBlockParams) -> 'TransformerBlock':
        """ Create a TransformerBlock instance from saved parameters. """

        instance = cls()
        instance.ln1 = LayerNorm.from_params(params.ln1)
        instance.self_attention = MultiHeadAttention.from_params(params.self_attention)
        instance.ln2 = LayerNorm.from_params(params.ln2)
        instance.mlp = MLP.from_params(params.mlp)
        return instance


    def forward(self, inputs : xp.ndarray, train: bool = True) -> xp.ndarray:
        """ Forward pass through the Transformer block. """

        # Layer normalization and residual connection for the self-attention sub-layer
        norm_x = self.ln1.forward(inputs)
        attn_out = self.self_attention.forward(norm_x)
        attn_out = self.dropout.forward(attn_out, train=train)
        x = inputs + attn_out
        x = self.dropout.forward(x, train=train)

        # Layer normalization and residual connection for the MLP sub-layer
        norm_x = self.ln2.forward(x)
        mlp_out = self.mlp.feed_forward(norm_x, train=train)
        x2 = x + mlp_out
        x2 = self.dropout.forward(x2, train=train)

        return x2


    def backward(self, grad_x2: xp.ndarray) -> xp.ndarray:
        """ Backward pass through the Transformer block. """

        # === 1. Dropout sur x2 ===
        grad_x2 = self.dropout.backward(grad_x2)

        # === 2. MLP ===
        grad_mlp_out = grad_x2                         # x2 = x + mlp_out
        grad_mlp = self.mlp.backward(grad_mlp_out)
        grad_ln2_input = self.ln2.backward(grad_mlp)
        grad_x = grad_ln2_input + grad_mlp_out         # skip connection

        # === 3. Dropout sur x ===
        grad_x = self.dropout.backward(grad_x)

        # === 4. Self-attention ===
        grad_attn_out = grad_x                         # x = inputs + attn_out
        grad_attn_out = self.dropout.backward(grad_attn_out)  # dropout sur attn_out
        grad_attn = self.self_attention.backward(grad_attn_out)
        grad_ln1_input = self.ln1.backward(grad_attn)

        # === 5. Skip connection initiale ===
        grad_input = grad_ln1_input + grad_attn_out

        return grad_input


    def step(self, lr : float):
        """ Update the parameters of the Transformer block using gradient descent. """

        self.self_attention.step(lr)
        self.mlp.step(lr)
        self.ln1.step(lr)
        self.ln2.step(lr)


    def zero_grad(self):
        """ Reset the gradients of the Transformer block parameters. """

        self.self_attention.zero_grad()
        self.mlp.zero_grad()
        self.ln1.zero_grad()
        self.ln2.zero_grad()
        self.dropout.zero_grad()


    def get_parameters(self) -> TransformerBlockParams:
        """ Return the parameters of the Transformer block for saving. """

        return TransformerBlockParams(
            ln1=self.ln1.get_params(),
            self_attention=self.self_attention.get_params(),
            ln2=self.ln2.get_params(),
            mlp=self.mlp.get_parameters()
        )