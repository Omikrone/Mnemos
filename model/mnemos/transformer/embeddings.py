from mnemos import xp
from mnemos.transformer.gradient import Param
from mnemos.config.params import EMBEDDING_DIM, MAX_SEQUENCE_LENGTH


class TokenEmbedding:
    """ Token Embedding class for converting token IDs to dense vectors. """

    vocab_size : int
    matrix : Param
    input_batches : xp.ndarray

    def __init__(self, vocab_size: int):
        """ Initialize the TokenEmbedding with the vocabulary size. """

        self.vocab_size = vocab_size

        # Random initialization of embedding vectors
        self.matrix = Param(xp.random.uniform(
            low=-0.1, high=0.1, size=(vocab_size, EMBEDDING_DIM)
        ))


    @classmethod
    def from_params(cls, embedding_matrix: xp.ndarray) -> 'TokenEmbedding':
        """ Create a TokenEmbedding instance from saved parameters. """

        instance = cls(vocab_size=embedding_matrix.shape[0])
        instance.matrix = Param(embedding_matrix)
        return instance


    def embed_batches(self, input_batches : xp.ndarray) -> xp.ndarray:
        """ Convert token IDs in input_batches to dense vectors. """

        self.input_batches = input_batches
        return self.matrix.value[input_batches]
    
    
    def backward(self, d_output: xp.ndarray) -> xp.ndarray:
        """
        d_output: gradient de shape (batch_size, seq_len, embedding_dim)
        """
        if self.matrix.gradient is None:
            self.matrix.gradient = xp.zeros_like(self.matrix.value)

        xp.add.at(
            self.matrix.gradient,
            self.input_batches,
            d_output
        )

        return xp.zeros_like(d_output)


    def zero_grad(self):
        """ Reset the gradient of the embedding matrix. """
        
        self.matrix.zero_grad()


class PositionEmbedding:
    """ Position Embedding class for adding positional information to token embeddings. """

    matrix : Param
    seq_len : int


    def __init__(self):
        # Random initialization of embedding vectors
        self.matrix = Param(xp.random.uniform(
            low=-0.1, high=0.1, size=(MAX_SEQUENCE_LENGTH, EMBEDDING_DIM)
        ))
    
    @classmethod
    def from_params(cls, position_matrix: xp.ndarray) -> 'PositionEmbedding':
        """ Create a PositionEmbedding instance from saved parameters. """

        instance = cls()
        instance.matrix = Param(position_matrix)
        return instance


    def embed_positions(self, batch_size: int, seq_len: int) -> xp.ndarray:
        """ Convert position indices to dense vectors. """

        if seq_len > MAX_SEQUENCE_LENGTH:
            raise ValueError(f"seq_len {seq_len} > max_length {MAX_SEQUENCE_LENGTH}")

        self.seq_len = seq_len  # to save for backward

        positions = self.matrix.value[:seq_len]
        return xp.broadcast_to(positions, (batch_size, seq_len, EMBEDDING_DIM))


    def backward(self, d_output: xp.ndarray) -> xp.ndarray:
        """
        d_output: gradient remontant depuis la suite du réseau,
                de shape (batch_size, seq_len, embedding_dim)
        """
        if self.matrix.gradient is None:
            self.matrix.gradient = xp.zeros_like(self.matrix.value)

        grad_sum = xp.sum(d_output, axis=0)

        self.matrix.gradient[:self.seq_len] += grad_sum

        return d_output
    

    def zero_grad(self):
        """ Reset the gradient of the position embedding matrix. """

        self.matrix.zero_grad()