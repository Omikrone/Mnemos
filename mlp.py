import numpy as np

from attention import add_attention
from embeddings import EMBEDDING_DIMENSION, create_total_embedding


HIDDEN_DIMENSION = 128  # Dimension cachée pour le MLP

W1 = np.random.randn(EMBEDDING_DIMENSION, HIDDEN_DIMENSION) * 0.01  # Poids de la première couche
W2 = np.random.randn(HIDDEN_DIMENSION, EMBEDDING_DIMENSION) * 0.01  # Poids de la seconde couche

b1 = np.zeros((1, HIDDEN_DIMENSION))  # Biais de la première couche
b2 = np.zeros((1, EMBEDDING_DIMENSION))  # Biais de la seconde couche


def MLP(inputs: np.ndarray) -> np.ndarray:
    """ Multilayer Perceptron (MLP) pour transformer les embeddings """
    
    h = np.dot(inputs, W1) + b1 # Produit matriciel avec la première couche
    h_relu = np.maximum(0, h)     # ReLU
    out = np.dot(h_relu, W2) + b2 # Produit matriciel avec la seconde couche
    
    return out


output = add_attention()
output = MLP(output)
print("Output shape:", output.shape)
print("Output:", output)