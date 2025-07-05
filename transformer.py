import numpy as np
from attention import add_attention
from data_retriever import create_batches
from embeddings import EMBEDDING_DIMENSION, create_total_embedding
from mlp import HIDDEN_DIMENSION, MLP


w1 = np.random.randn(EMBEDDING_DIMENSION, HIDDEN_DIMENSION) * 0.01  # Poids de la première couche
w2 = np.random.randn(HIDDEN_DIMENSION, EMBEDDING_DIMENSION) * 0.01  # Poids de la seconde couche

b1 = np.zeros((1, HIDDEN_DIMENSION))  # Biais de la première couche
b2 = np.zeros((1, EMBEDDING_DIMENSION))  # Biais de la seconde couche

W_out = np.random.randn(EMBEDDING_DIMENSION, 85) * 0.02
b_out = np.zeros((85,))


def transformer_block(inputs: np.ndarray) -> np.ndarray:
    
    attn_out = add_attention(inputs)
    inputs += attn_out

    mlp_out = MLP(inputs, w1, b1, w2, b2)
    inputs += mlp_out

    return inputs


def softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))  # stabilité numérique
    return e_x / np.sum(e_x, axis=-1, keepdims=True)

def cross_entropy_loss(logits, targets):
    probs = softmax(logits)

    batch_size, seq_len = targets.shape
    probs_cible = probs[np.arange(batch_size)[:, None], np.arange(seq_len), targets]

    loss = -np.log(probs_cible + 1e-9)
    
    return np.mean(loss)


def pipeline():
    input_batches, target_batches = create_batches()
    print("Input batches shape:", input_batches)
    embedded_inputs = create_total_embedding(input_batches, target_batches)
    out = transformer_block(embedded_inputs)
    
    logits = out @ W_out + b_out
    predicted_tokens = np.argmax(logits, axis=-1)

    loss = cross_entropy_loss(logits, target_batches)
    print("Loss:", loss)


if __name__ == "__main__":
    pipeline()