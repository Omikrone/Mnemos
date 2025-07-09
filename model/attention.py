import numpy as np

from model.embeddings import EMBEDDING_DIMENSION


class Attention:

    query_matrix : np.ndarray
    key_matrix : np.ndarray
    value_matrix : np.ndarray

    def __init__(self):

        # Initialisation alétoire des matrices de l'attention
        self.query_matrix = np.random.randn(EMBEDDING_DIMENSION, EMBEDDING_DIMENSION)
        self.key_matrix = np.random.randn(EMBEDDING_DIMENSION, EMBEDDING_DIMENSION)
        self.value_matrix = np.random.randn(EMBEDDING_DIMENSION, EMBEDDING_DIMENSION)


    def add_attention(self, input_embeddings) -> np.ndarray:
        """ Add attention weights to the inputs """

        # Matrix product to get Query, Key and Value matrices
        query = input_embeddings @ self.query_matrix
        key = input_embeddings @ self.key_matrix
        value = input_embeddings @ self.value_matrix

        scores = query @ key / np.sqrt(EMBEDDING_DIMENSION)
        
        # 4. Softmax ligne par ligne pour normaliser
        attention_weights = np.exp(scores)
        attention_weights /= np.sum(attention_weights, axis=1, keepdims=True)

        # On applique les poids d'attention aux valeurs
        output = attention_weights @ value
        return output