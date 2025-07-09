import numpy as np

from model.embeddings import PositionEmbedding, TokenEmbedding
from model.transformer_block import TransformerBlock


class TransformerModel:

    embedding : TokenEmbedding
    position : PositionEmbedding
    block : TransformerBlock


    def __init__(self, vocab_size : int):
        self.embedding = TokenEmbedding(vocab_size)
        self.position = PositionEmbedding()
        self.block = TransformerBlock(vocab_size)


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