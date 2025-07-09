import numpy as np

from model.gradient import Param


EMBEDDING_DIMENSION = 64
MAX_SEQUENCE_LENGTH = 64
BATCH_SIZE = 8


class TokenEmbedding:

    vocab_size : int
    matrix : Param
    input_batches : np.ndarray

    def __init__(self, vocab_size: int):
        self.vocab_size = vocab_size
        # Initialisation aléatoire uniforme des vecteurs d'embedding
        self.matrix = Param(np.random.uniform(
            low=-0.1, high=0.1, size=(vocab_size, EMBEDDING_DIMENSION)
        ))

    @classmethod
    def from_params(cls, embedding_matrix: np.ndarray) -> 'TokenEmbedding':
        instance = cls(vocab_size=embedding_matrix.shape[0])
        instance.matrix = Param(embedding_matrix)
        return instance

    def embed_batches(self, input_batches : np.ndarray) -> np.ndarray:
        self.input_batches = input_batches
        return self.matrix.value[input_batches]
    
    def backward(self, d_output: np.ndarray):

        # Initialiser le gradient si ce n'est pas fait
        if self.matrix.gradient is None:
            self.matrix.gradient = np.zeros_like(self.matrix.value)

        # Pour chaque position dans le batch, accumuler le gradient sur la bonne ligne de la matrice
        batch_size, seq_len = self.input_batches.shape
        for i in range(batch_size):
            for j in range(seq_len):
                token_id = self.input_batches[i, j]
                self.matrix.gradient[token_id] += d_output[i, j]
        return self.matrix.gradient
    
    def zero_grad(self):
        self.matrix.zero_grad()


class PositionEmbedding:

    matrix : Param
    seq_len : int

    def __init__(self):
        # Initialisation aléatoire uniforme des vecteurs d'embedding
        self.matrix = Param(np.random.uniform(
            low=-0.1, high=0.1, size=(MAX_SEQUENCE_LENGTH, EMBEDDING_DIMENSION)
        ))
    
    @classmethod
    def from_params(cls, position_matrix: np.ndarray) -> 'PositionEmbedding':
        instance = cls()
        instance.matrix = Param(position_matrix)
        return instance

    def embed_positions(self, batch_size: int, seq_len: int) -> np.ndarray:
        if seq_len > MAX_SEQUENCE_LENGTH:
            raise ValueError(f"seq_len {seq_len} > max_length {MAX_SEQUENCE_LENGTH}")

        self.seq_len = seq_len  # à sauvegarder pour le backward

        positions = self.matrix.value[:seq_len]
        return np.broadcast_to(positions, (batch_size, seq_len, EMBEDDING_DIMENSION))

    def backward(self, d_output: np.ndarray):
        if self.matrix.gradient is None:
            self.matrix.gradient = np.zeros_like(self.matrix.value)

        grad_sum = np.sum(d_output, axis=0) 

        self.matrix.gradient[:self.seq_len] += grad_sum
        return self.matrix.gradient
    
    def zero_grad(self):
        self.matrix.zero_grad()