import numpy as np

from model.embeddings import PositionEmbedding, TokenEmbedding
from model.save_model import ModelParams
from model.transformer_block import TransformerBlock


class TransformerModel:
    """ Transformer model class that combines token and position embeddings with a Transformer block. """

    embedding : TokenEmbedding
    position : PositionEmbedding
    block : TransformerBlock


    def __init__(self, vocab_size : int):
        """ Initialize the Transformer model with token and position embeddings and a Transformer block. """

        self.embedding = TokenEmbedding(vocab_size)
        self.position = PositionEmbedding()
        self.block = TransformerBlock(vocab_size)


    @classmethod
    def from_params(cls, params: ModelParams) -> 'TransformerModel':
        """ Create an instance of TransformerModel from saved parameters. """

        instance = cls(vocab_size=params.embedding_matrix.shape[0])
        instance.embedding = TokenEmbedding.from_params(params.embedding_matrix)
        instance.position = PositionEmbedding.from_params(params.position_matrix)
        instance.block = TransformerBlock.from_params(params.transformer_block_params)
        return instance


    def forward(self, inputs : np.ndarray) -> np.ndarray:
        """ Forward pass through the Transformer model. """

        batch_size, seq_len = inputs.shape
        token_embeds = self.embedding.embed_batches(inputs)
        pos_embeds = self.position.embed_positions(batch_size, seq_len)
        x = token_embeds + pos_embeds

        logits = self.block.forward(x)
        return logits
    
    
    def backward(self, loss_gradient) -> np.ndarray:
        """ Backward pass through the Transformer model. """

        grad = self.block.backward(loss_gradient)
        grad = self.position.backward(grad)
        grad = self.embedding.backward(grad)
        return grad
    

    def sample_top_k(self, logits, k=10) -> int:
        """ Sample from the top k logits using softmax. """

        top_k_indices = np.argpartition(logits, -k)[-k:]
        top_k_logits = logits[top_k_indices]
        probs = np.exp(top_k_logits) / np.sum(np.exp(top_k_logits))
        return np.random.choice(top_k_indices, p=probs)


    def predict_next_token(self, input_ids: np.ndarray) -> int:
        """ Predict the next token given the input IDs. """

        logits = self.forward(input_ids)
        last_logits = logits[0, -1]
        prediction = self.sample_top_k(last_logits, k=10)
        #prediction = np.argmax(last_logits)
        return prediction
    

    def step(self, lr : float):
        """ Update the parameters of the Transformer model using gradient descent. """

        self.embedding.matrix.step(lr)
        self.position.matrix.step(lr)
        self.block.step(lr)


    def zero_grad(self):
        """ Reset the gradients of the Transformer model parameters. """

        self.embedding.zero_grad()
        self.position.zero_grad()
        self.block.zero_grad()


    def get_parameters(self) -> ModelParams:
        """ Return the parameters of the Transformer model for saving. """

        return ModelParams(
            embedding_matrix=self.embedding.matrix.value,
            position_matrix=self.position.matrix.value,
            transformer_block_params=self.block.get_parameters()
        )