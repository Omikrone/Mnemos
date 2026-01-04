import numpy as np

from mnemos.transformer.gradient import Param
from mnemos.config.params import EMBEDDING_DIM, MAX_SEQUENCE_LENGTH


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
            low=-0.1, high=0.1, size=(vocab_size, EMBEDDING_DIM)
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
    
    
    def backward(self, d_output: np.ndarray) -> np.ndarray:
        """
        d_output: gradient de shape (batch_size, seq_len, embedding_dim)
        """
        # Initialisation du gradient s'il n'existe pas
        if self.matrix.gradient is None:
            self.matrix.gradient = np.zeros_like(self.matrix.value)

        # Accumulation vectorisée des gradients pour chaque token
        # self.input_batches shape = (batch_size, seq_len)
        # d_output shape = (batch_size, seq_len, embedding_dim)
        np.add.at(
            self.matrix.gradient,                   # accumulation dans la matrice
            self.input_batches,                     # indices (batch_size, seq_len)
            d_output                                # gradients correspondants
        )

        # Pour les IDs, on ne peut pas propager un vrai gradient en amont,
        # on renvoie un tensor de zéros de même shape que d_output.
        return np.zeros_like(d_output)


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
            low=-0.1, high=0.1, size=(MAX_SEQUENCE_LENGTH, EMBEDDING_DIM)
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
        return np.broadcast_to(positions, (batch_size, seq_len, EMBEDDING_DIM))


    def backward(self, d_output: np.ndarray) -> np.ndarray:
        """
        d_output: gradient remontant depuis la suite du réseau,
                de shape (batch_size, seq_len, embedding_dim)
        """
        # Initialisation du gradient si besoin
        if self.matrix.gradient is None:
            self.matrix.gradient = np.zeros_like(self.matrix.value)

        # Somme des gradients sur le batch pour chaque position
        # grad_sum[i] = somme_{batch, dim} d_output[batch, i, :]
        grad_sum = np.sum(d_output, axis=0)  # shape = (seq_len, embedding_dim)

        # On ajoute ce gradient aux lignes correspondantes de la matrice de pos.
        # Attention à n'accumuler QUE sur la longueur seq_len actuelle.
        self.matrix.gradient[:self.seq_len] += grad_sum

        # On renvoie le gradient en amont pour les embeddings de tokens
        # (ici, c'est exactement d_output, pas besoin de le modifier)
        return d_output
    

    def zero_grad(self):
        """ Reset the gradient of the position embedding matrix. """

        self.matrix.zero_grad()