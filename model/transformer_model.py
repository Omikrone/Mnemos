import numpy as np

from model.embeddings import PositionEmbedding, TokenEmbedding
from model.gradient import Param
from model.save_model import ModelParams
from model.transformer_block import TransformerBlock
from config.params import EMBEDDING_DIM, NB_LAYERS


class TransformerModel:
    """ Transformer model class that combines token and position embeddings with a Transformer block. """

    embedding : TokenEmbedding
    position : PositionEmbedding
    blocks : list[TransformerBlock]
    w_out : Param
    b_out : Param


    def __init__(self, vocab_size : int):
        """ Initialize the Transformer model with token and position embeddings and a Transformer block. """

        self.embedding = TokenEmbedding(vocab_size)
        self.position = PositionEmbedding()
        self.blocks = [TransformerBlock() for _ in range(NB_LAYERS)]
        self.w_out = Param(np.random.randn(EMBEDDING_DIM, vocab_size) * 0.02)
        self.b_out = Param(np.zeros((vocab_size,)))


    @classmethod
    def from_params(cls, params: ModelParams) -> 'TransformerModel':
        """ Create an instance of TransformerModel from saved parameters. """

        instance = cls(vocab_size=params.embedding_matrix.shape[0])
        instance.embedding = TokenEmbedding.from_params(params.embedding_matrix)
        instance.position = PositionEmbedding.from_params(params.position_matrix)
        instance.blocks = [
            TransformerBlock.from_params(block_params) for block_params in params.transformer_block_params
        ]
        instance.w_out = Param(params.w_out)
        instance.b_out = Param(params.b_out)
        return instance


    def forward(self, inputs : np.ndarray) -> np.ndarray:
        """ Forward pass through the Transformer model. """

        batch_size, seq_len = inputs.shape
        token_embeds = self.embedding.embed_batches(inputs)
        pos_embeds = self.position.embed_positions(batch_size, seq_len)
        x = token_embeds + pos_embeds

        for block in self.blocks:
            # Forward pass through each Transformer block
            x = block.forward(x)
        self.last_x = x
        
        # Final linear transformation to produce logits
        logits = x @ self.w_out.value + self.b_out.value

        return logits


    def backward(self, loss_gradient) -> np.ndarray:
        """ Backward pass through the Transformer model. """

        # Compute gradients for the output layer
        self.w_out.gradient = np.einsum('btd,btv->dv', self.last_x, loss_gradient)
        self.b_out.gradient = np.sum(loss_gradient, axis=(0, 1))

        loss_gradient = loss_gradient @ self.w_out.value.T

        for block in reversed(self.blocks):
            loss_gradient = block.backward(loss_gradient)
        grad_embedding = self.embedding.backward(loss_gradient)
        grad_position = self.position.backward(loss_gradient)
        return grad_position + grad_embedding
    

    def step(self, lr : float):
        """ Update the parameters of the Transformer model using gradient descent. """

        self.embedding.matrix.step(lr)
        self.position.matrix.step(lr)
        for block in self.blocks:
            block.step(lr)
        self.w_out.step(lr)
        self.b_out.step(lr)


    def zero_grad(self):
        """ Reset the gradients of the Transformer model parameters. """

        self.embedding.zero_grad()
        self.position.zero_grad()
        for block in self.blocks:
            block.zero_grad()
        self.w_out.zero_grad()
        self.b_out.zero_grad()


    def get_parameters(self) -> ModelParams:
        """ Return the parameters of the Transformer model for saving. """

        return ModelParams(
            embedding_matrix=self.embedding.matrix.value,
            position_matrix=self.position.matrix.value,
            transformer_block_params= [block.get_parameters() for block in self.blocks],
            w_out=self.w_out.value,
            b_out=self.b_out.value
        )