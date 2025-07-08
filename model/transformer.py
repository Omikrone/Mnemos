import numpy as np
from attention import add_attention
from training.data_retriever import create_batches
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


def cross_entropy_loss(predicted_class, goal_class) -> float:
    """ Calcule la perte d'entropie croisée entre la class prédite et la classe réelle """

    for prob_i in range(goal_class):

        # Dans le cas d'un vecteur one-hot, la perte d'entropie croisée se résume à L = -log(y')
        # Avec y' la probabilité prédite du bon token
        if goal_class[prob_i] == 1:
            return -np.log(predicted_class[prob_i])


def pipeline():
    input_batches, target_batches = create_batches()
    print("Input batches shape:", input_batches)
    embedded_inputs = create_total_embedding(input_batches, target_batches)
    out = transformer_block(embedded_inputs)
    
    logits = out @ W_out + b_out
    probs_in = softmax(logits)
    print(probs_in.shape)

    #loss = cross_entropy_loss()


def gradient_descent(loss : int, logits : np.ndarray, target_batches : np.ndarray):
    probs_in = softmax(logits)

    batch_size, seq_len = target_batches.shape
    probs_cible = probs_in[np.arange(batch_size)[:, None], np.arange(seq_len), target_batches]
    gradient_logits(probs_in, probs_cible)

        

def gradient_logits(probs_in, probs_cible):
    print(probs_cible)
    input()
    return probs_in - probs_cible # Provient de la dérivée partielle de la perte par rapport au logit


if __name__ == "__main__":
    pipeline()