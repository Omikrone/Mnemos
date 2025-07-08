import numpy as np

from attention import add_attention
from embeddings import EMBEDDING_DIMENSION, create_total_embedding


HIDDEN_DIMENSION = 128  # Dimension cachée pour le MLP




def MLP(inputs: np.ndarray, W1, b1, W2, b2) -> np.ndarray:
    """ Multilayer Perceptron (MLP) pour transformer les embeddings """
    
    h = np.dot(inputs, W1) + b1 # Produit matriciel avec la première couche
    print("Shape of h:", h.shape)  # Affichage de la forme de h pour le débogage
    h_relu = np.maximum(0, h)     # ReLU
    out = np.dot(h_relu, W2) + b2 # Produit matriciel avec la seconde couche
    print("Shape of out:", out.shape)  # Affichage de la forme de out pour le débogage
    
    return out