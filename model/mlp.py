import numpy as np

from model.embeddings import EMBEDDING_DIMENSION

HIDDEN_DIMENSION = 128  # Dimension cachée pour le MLP


class MLP:

    w1 : np.ndarray
    b1 : np.ndarray
    w2 : np.ndarray
    b2 : np.ndarray

    def __init__(self):

        # Initialisation aléatoire des poids de la première et seconde couche
        self.w1 = np.random.randn(EMBEDDING_DIMENSION, HIDDEN_DIMENSION) * 0.01  # Poids de la première couche
        self.w2 = np.random.randn(HIDDEN_DIMENSION, EMBEDDING_DIMENSION) * 0.01  # Poids de la seconde couche

        # Initialisation à 0 des biais
        self.b1 = np.zeros((1, HIDDEN_DIMENSION))  # Biais de la première couche
        self.b2 = np.zeros((1, EMBEDDING_DIMENSION))  # Biais de la seconde couche

    def feed_forward(self, inputs : np.ndarray) -> np.ndarray:
        h = inputs @ self.w1 + self.b1
        h_relu = np.maximum(0, h)
        out = h_relu @ self.w2 + self.b2

        return out