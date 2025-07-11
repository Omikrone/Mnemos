import numpy as np


class CrossEntropyLoss:

    softmax_value : float
    logits : np.ndarray
    targets : np.ndarray

    def __init__(self):
        self.softmax_value = None
        self.logits = None
        self.targets = None

    def _softmax(self, logits: np.ndarray) -> np.ndarray:
        """ Normalise les logits pour en faire une distribution de probabilité. """

        e_x = np.exp(logits - np.max(logits, axis=-1, keepdims=True))  # stabilité numérique
        return e_x / np.sum(e_x, axis=-1, keepdims=True)


    def __call__(self, logits: np.ndarray, targets: np.ndarray) -> float:
        """ Calcule la perte d'entropie croisée entre les valeurs prédites et les cibles. """

        self.logits = logits
        self.targets = targets

        # On applique softmax aux logits pour obtenir les probabilités
        self.softmax_value = self._softmax(logits)

        # On extrait les probabilités des tokens cibles
        batch_size, seq_len, _ = logits.shape
        batch_indices = np.arange(batch_size)[:, None]
        seq_indices = np.arange(seq_len)[None, :]
        correct_token_probs = self.softmax_value[batch_indices, seq_indices, targets]

        # On calcule la perte d'entropie croisée = -log(probabilité correcte)
        loss = -np.log(correct_token_probs + 1e-9).mean()
        return loss
    

    def backward(self, logits: np.ndarray, targets: np.ndarray) -> float:
        """ Calcule le gradient de la perte par rapport aux valeurs prédites. """
        
        batch_size, seq_len, _ = logits.shape
        predicted_probs = self._softmax(logits)

        # On forme un vecteur one-hot des cibles pour le calcul du gradient
        real_class = np.zeros_like(predicted_probs)
        batch_indices = np.arange(batch_size)[:, None]
        seq_indices = np.arange(seq_len)[None, :]
        real_class[batch_indices, seq_indices, targets] = 1

        # Calcul du gradient de la perte -> dérivée partielle de la perte par rapport aux logits
        grad = (predicted_probs - real_class) / (batch_size * seq_len)
        return grad