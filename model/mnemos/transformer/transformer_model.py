from mnemos import xp
from mnemos.transformer.embeddings import PositionEmbedding, TokenEmbedding
from mnemos.transformer.gradient import Param
from mnemos.transformer.save_model import ModelParams
from mnemos.transformer.transformer_block import TransformerBlock
from mnemos.config.params import EMBEDDING_DIM, NB_LAYERS


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
        self.w_out = Param(xp.random.randn(EMBEDDING_DIM, vocab_size) * 0.02)
        self.b_out = Param(xp.zeros((vocab_size,)))


    @classmethod
    def from_params(cls, params: dict) -> 'TransformerModel':
        """ Create an instance of TransformerModel from saved parameters. """

        model_params = ModelParams.from_state_dict(params)

        # Load parameters into an instance of ModelParams from dict
        instance = cls(vocab_size=model_params.embedding_matrix.shape[0])
        instance.embedding = TokenEmbedding.from_params(model_params.embedding_matrix)
        instance.position = PositionEmbedding.from_params(model_params.position_matrix)
        instance.blocks = [
            TransformerBlock.from_params(block_params) for block_params in model_params.transformer_block_params
        ]
        instance.w_out = Param(model_params.w_out)
        instance.b_out = Param(model_params.b_out)
        return instance


    def forward(self, inputs : xp.ndarray, train: bool = True) -> xp.ndarray:
        """ Forward pass through the Transformer model. """

        batch_size, seq_len = inputs.shape
        token_embeds = self.embedding.embed_batches(inputs)
        pos_embeds = self.position.embed_positions(batch_size, seq_len)
        x = token_embeds + pos_embeds

        for block in self.blocks:
            # Forward pass through each Transformer block
            x = block.forward(x, train=train)
        self.last_x = x
        
        # Final linear transformation to produce logits
        logits = x @ self.w_out.value + self.b_out.value

        return logits


    def backward(self, loss_gradient) -> xp.ndarray:
        """ Backward pass through the Transformer model. """

        # Compute gradients for the output layer
        self.w_out.gradient = xp.einsum('btd,btv->dv', self.last_x, loss_gradient)
        self.b_out.gradient = xp.sum(loss_gradient, axis=(0, 1))

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