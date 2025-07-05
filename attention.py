import numpy as np

from embeddings import EMBEDDING_DIMENSION, create_total_embedding

# Matrices de poids pour les transformations Q, K, V (initialisation aléatoire)
W_Q = np.random.randn(EMBEDDING_DIMENSION, EMBEDDING_DIMENSION)
W_K = np.random.randn(EMBEDDING_DIMENSION, EMBEDDING_DIMENSION)
W_V = np.random.randn(EMBEDDING_DIMENSION, EMBEDDING_DIMENSION)


def add_attention(input_embeddings) -> np.ndarray:
    """ Add attention weights to the inputs """

    # Matrix product to get Query, Key and Value matrices
    Q = input_embeddings @ W_Q
    K = input_embeddings @ W_K
    V = input_embeddings @ W_V

    scores = Q @ K / np.sqrt(EMBEDDING_DIMENSION)
    print("Scores shape:", scores.shape)
    
    # 4. Softmax ligne par ligne pour normaliser
    attention_weights = np.exp(scores)
    attention_weights /= np.sum(attention_weights, axis=1, keepdims=True)

    # On applique les poids d'attention aux valeurs
    print("Attention weights shape:", attention_weights.shape)
    print("Value matrix shape:", V.shape)
    output = attention_weights @ V
    return output