import numpy as np


EMBEDDING_DIMENSION = 64
MAX_SEQUENCE_LENGTH = 64
BATCH_SIZE = 8


class TokenEmbedding:

    vocab_size : int
    matrix : np.ndarray | list

    def __init__(self, vocab_size: int):
        self.vocab_size = vocab_size
        # Initialisation aléatoire uniforme des vecteurs d'embedding
        self.matrix: np.ndarray = np.random.uniform(
            low=-0.1, high=0.1, size=(vocab_size, EMBEDDING_DIMENSION)
        )

    def embed_batches(self, input_batches : np.ndarray) -> np.ndarray:
        return self.matrix[input_batches]


class PositionEmbedding:

    matrix : np.ndarray | list

    def __init__(self):
        # Initialisation aléatoire uniforme des vecteurs d'embedding
        self.matrix: np.ndarray = np.random.uniform(
            low=-0.1, high=0.1, size=(MAX_SEQUENCE_LENGTH, EMBEDDING_DIMENSION)
        )

    def embed_positions(self, batch_size : int, seq_len : int) -> np.ndarray:

        if seq_len > MAX_SEQUENCE_LENGTH:
            raise ValueError(f"seq_len {seq_len} > max_length {MAX_SEQUENCE_LENGTH}")
        
        positions = self.matrix[:seq_len]
        return np.broadcast_to(positions, (batch_size, seq_len, EMBEDDING_DIMENSION))