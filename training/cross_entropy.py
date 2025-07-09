import numpy as np


class CrossEntropyLoss:

    softmax_value : float
    logits : np.ndarray
    targets : np.ndarray

    def __init__(self):
        self.softmax_value = None
        self.logits = None
        self.targets = None

    def _softmax(self, x) -> np.ndarray:
        e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))  # stabilité numérique
        return e_x / np.sum(e_x, axis=-1, keepdims=True)


    def __call__(self, logits: np.ndarray, targets: np.ndarray) -> float:
        """
        logits : (batch, seq_len, vocab_size)
        targets : (batch, seq_len)  → indices des bons tokens
        """
        self.logits = logits
        self.targets = targets
        self.softmax_value = self._softmax(logits)

        batch_size, seq_len, _ = logits.shape

        batch_indices = np.arange(batch_size)[:, None]
        seq_indices = np.arange(seq_len)[None, :]

        correct_token_probs = self.softmax_value[batch_indices, seq_indices, targets]

        # Perte d'entropie croisée = -log(probabilité correcte)
        loss = -np.log(correct_token_probs + 1e-9).mean()

        return loss
    

    def backward(self, logits: np.ndarray, targets: np.ndarray) -> float:
        batch_size, seq_len, _ = logits.shape
        softmax = self._softmax(logits)

        one_hot = np.zeros_like(softmax)
        batch_indices = np.arange(batch_size)[:, None]
        seq_indices = np.arange(seq_len)[None, :]
        one_hot[batch_indices, seq_indices, targets] = 1

        grad = (softmax - one_hot) / (batch_size * seq_len)
        return grad