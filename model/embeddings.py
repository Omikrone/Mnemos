import numpy as np

from model.gradient import Param


EMBEDDING_DIMENSION = 64
MAX_SEQUENCE_LENGTH = 64
BATCH_SIZE = 8


class TokenEmbedding:
    """ Token Embedding class for converting token IDs to dense vectors. """

    vocab_size : int
    matrix : Param
    input_batches : np.ndarray

    def __init__(self, vocab_size: int):
        """ Initialize the TokenEmbedding with the vocabulary size. """

        self.vocab_size = vocab_size

        # Random initialization of embedding vectors
        self.matrix = Param(np.random.uniform(
            low=-0.1, high=0.1, size=(vocab_size, EMBEDDING_DIMENSION)
        ))


    @classmethod
    def from_params(cls, embedding_matrix: np.ndarray) -> 'TokenEmbedding':
        """ Create a TokenEmbedding instance from saved parameters. """

        instance = cls(vocab_size=embedding_matrix.shape[0])
        instance.matrix = Param(embedding_matrix)
        return instance


    def embed_batches(self, input_batches : np.ndarray) -> np.ndarray:
        """ Convert token IDs in input_batches to dense vectors. """

        self.input_batches = input_batches
        return self.matrix.value[input_batches]
    
    
    def backward(self, d_output: np.ndarray):
        """ Compute the gradients of the embedding matrix for backpropagation. """

        # Initialize gradient if not done
        if self.matrix.gradient is None:
            self.matrix.gradient = np.zeros_like(self.matrix.value)

        # For each position in the batch, accumulate the gradient on the correct row of the matrix
        batch_size, seq_len = self.input_batches.shape
        for i in range(batch_size):
            for j in range(seq_len):
                token_id = self.input_batches[i, j]
                self.matrix.gradient[token_id] += d_output[i, j]
        return self.matrix.gradient


    def zero_grad(self):
        """ Reset the gradient of the embedding matrix. """
        
        self.matrix.zero_grad()


class PositionEmbedding:
    """ Position Embedding class for adding positional information to token embeddings. """

    matrix : Param
    seq_len : int


    def __init__(self):
        # Random initialization of embedding vectors

        self.matrix = Param(np.random.uniform(
            low=-0.1, high=0.1, size=(MAX_SEQUENCE_LENGTH, EMBEDDING_DIMENSION)
        ))
    
    @classmethod
    def from_params(cls, position_matrix: np.ndarray) -> 'PositionEmbedding':
        """ Create a PositionEmbedding instance from saved parameters. """

        instance = cls()
        instance.matrix = Param(position_matrix)
        return instance


    def embed_positions(self, batch_size: int, seq_len: int) -> np.ndarray:
        """ Convert position indices to dense vectors. """

        if seq_len > MAX_SEQUENCE_LENGTH:
            raise ValueError(f"seq_len {seq_len} > max_length {MAX_SEQUENCE_LENGTH}")

        self.seq_len = seq_len  # to save for backward

        positions = self.matrix.value[:seq_len]
        return np.broadcast_to(positions, (batch_size, seq_len, EMBEDDING_DIMENSION))


    def backward(self, d_output: np.ndarray):
        """ Compute the gradients of the position embedding matrix for backpropagation. """

        if self.matrix.gradient is None:
            self.matrix.gradient = np.zeros_like(self.matrix.value)

        grad_sum = np.sum(d_output, axis=0) 

        self.matrix.gradient[:self.seq_len] += grad_sum
        return self.matrix.gradient
    

    def zero_grad(self):
        """ Reset the gradient of the position embedding matrix. """

        self.matrix.zero_grad()