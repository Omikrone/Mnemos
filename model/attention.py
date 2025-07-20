import numpy as np

from model.gradient import Param
from model.save_model import AttentionParams
from params import EMBEDDING_DIM


class Attention:
    """ Attention class for implementing the attention mechanism in the Transformer model. """

    input_embeddings : np.ndarray
    query_matrix : Param
    key_matrix : Param
    value_matrix : Param
    output : np.ndarray
    attention_weights : np.ndarray
    value : np.ndarray
    key : np.ndarray
    query : np.ndarray


    def __init__(self):
        """ Initialize the attention matrices with random values. """

        # Random initialization of attention matrices
        self.query_matrix = Param(np.random.randn(EMBEDDING_DIM, EMBEDDING_DIM))
        self.key_matrix = Param(np.random.randn(EMBEDDING_DIM, EMBEDDING_DIM))
        self.value_matrix = Param(np.random.randn(EMBEDDING_DIM, EMBEDDING_DIM))


    @classmethod
    def from_params(cls, params: AttentionParams):
        """ Create an Attention instance from saved parameters. """

        instance = cls()
        instance.query_matrix = Param(params.query_matrix)
        instance.key_matrix = Param(params.key_matrix)
        instance.value_matrix = Param(params.value_matrix)
        return instance


    def add_attention(self, input_embeddings) -> np.ndarray:
        """ Add attention weights to the inputs """

        # Matrix product to get Query, Key and Value matrices
        self.input_embeddings = input_embeddings
        self.query = input_embeddings @ self.query_matrix.value
        self.key = input_embeddings @ self.key_matrix.value
        self.value = input_embeddings @ self.value_matrix.value

        scores = self.query @ self.key.transpose(0, 2, 1) / np.sqrt(EMBEDDING_DIM)

        # Softmax to get attention weights
        exp_scores = np.exp(scores - np.max(scores, axis=-1, keepdims=True))  # stab numérique
        attention_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
        self.attention_weights = attention_weights

        # Apply attention weights to the values
        self.output = self.attention_weights @ self.value
        return self.output
    

    def backward(self, d_output) -> np.ndarray:
        B, T, D = d_output.shape
        sqrt_d = np.sqrt(D)

        # Gradient with respect to value
        d_value = self.attention_weights.transpose(0, 2, 1) @ d_output  # (B, T, D)

        # Gradient with respect to attention_weights
        d_weights = d_output @ self.value.transpose(0, 2, 1)  # (B, T, T)

        # Backpropagation through softmax (stabilized)
        softmax = self.attention_weights
        d_scores = d_weights * softmax * (1 - softmax)

        # Gradient with respect to query and key
        d_query = d_scores @ self.key / sqrt_d
        d_key = d_scores.transpose(0, 2, 1) @ self.query / sqrt_d

        # Gradient with respect to weights
        self.query_matrix.gradient += self.input_embeddings.reshape(B*T, D).T @ d_query.reshape(B*T, D)
        self.key_matrix.gradient += self.input_embeddings.reshape(B*T, D).T @ d_key.reshape(B*T, D)
        self.value_matrix.gradient += self.input_embeddings.reshape(B*T, D).T @ d_value.reshape(B*T, D)

        # Gradient with respect to input embeddings
        d_input_q = d_query @ self.query_matrix.value.T
        d_input_k = d_key @ self.key_matrix.value.T
        d_input_v = d_value @ self.value_matrix.value.T

        return d_input_q + d_input_k + d_input_v
    

    def step(self, lr : float):
        """ Update the parameters of the attention mechanism with gradient descent. """

        self.query_matrix.step(lr)
        self.value_matrix.step(lr)
        self.key_matrix.step(lr)


    def zero_grad(self):
        """ Reset the gradients of the attention parameters. """

        self.query_matrix.zero_grad()
        self.value_matrix.zero_grad()
        self.key_matrix.zero_grad()
        self.attention_weights = None
        self.output = None
        self.input_embeddings = None
        self.query = None
        self.key = None
        self.value = None


    def get_parameters(self):
        """ Return the parameters of the attention mechanism for saving. """

        return AttentionParams(
            query_matrix=self.query_matrix.value,
            key_matrix=self.key_matrix.value,
            value_matrix=self.value_matrix.value
        )