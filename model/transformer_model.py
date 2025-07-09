import numpy as np

from model.embeddings import PositionEmbedding, TokenEmbedding
from model.save_model import ModelParams
from model.transformer_block import TransformerBlock


class TransformerModel:

    embedding : TokenEmbedding
    position : PositionEmbedding
    block : TransformerBlock


    def __init__(self, vocab_size : int):
        self.embedding = TokenEmbedding(vocab_size)
        self.position = PositionEmbedding()
        self.block = TransformerBlock(vocab_size)

    @classmethod
    def from_params(cls, params: ModelParams) -> 'TransformerModel':
        instance = cls(vocab_size=params.embedding_matrix.shape[0])
        instance.embedding = TokenEmbedding.from_params(params.embedding_matrix)
        instance.position = PositionEmbedding.from_params(params.position_matrix)
        instance.block = TransformerBlock.from_params(params.transformer_block_params)
        return instance


    def forward(self, inputs : np.ndarray) -> np.ndarray:
        batch_size, seq_len = inputs.shape
        token_embeds = self.embedding.embed_batches(inputs)
        pos_embeds = self.position.embed_positions(batch_size, seq_len)
        x = token_embeds + pos_embeds

        logits = self.block.forward(x)
        return logits
    

    def backward(self, loss_gradient) -> np.ndarray:
        grad = self.block.backward(loss_gradient)
        grad = self.position.backward(grad)
        grad = self.embedding.backward(grad)
        return grad
    
    def step(self, lr : float):
        self.embedding.matrix.step(lr)
        self.position.matrix.step(lr)
        self.block.step(lr)

    def zero_grad(self):
        self.embedding.zero_grad()
        self.position.zero_grad()
        self.block.zero_grad()

    def get_parameters(self):
        return ModelParams(
            embedding_matrix=self.embedding.matrix.value,
            position_matrix=self.position.matrix.value,
            transformer_block_params=self.block.get_parameters()
        )