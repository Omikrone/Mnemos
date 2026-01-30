from mnemos import xp


class CrossEntropyLoss:
    """ CrossEntropyLoss class for computing the cross-entropy loss and its gradient. """

    softmax_value : float
    logits : xp.ndarray
    targets : xp.ndarray


    def __init__(self):
        self.softmax_value = None
        self.logits = None
        self.targets = None


    def _softmax(self, logits: xp.ndarray) -> xp.ndarray:
        """ Normalize logits to probabilities using softmax. """

        e_x = xp.exp(logits - xp.max(logits, axis=-1, keepdims=True))  # stabilité numérique
        return e_x / xp.sum(e_x, axis=-1, keepdims=True)


    def __call__(self, logits: xp.ndarray, targets: xp.ndarray) -> float:
        """ Compute the cross-entropy loss between the predicted values and the targets. """

        self.logits = logits
        self.targets = targets

        # Apply softmax to the logits to get the probabilities
        self.softmax_value = self._softmax(logits)

        # Extract the probabilities of the target tokens
        batch_size, seq_len, _ = logits.shape
        batch_indices = xp.arange(batch_size)[:, None]
        seq_indices = xp.arange(seq_len)[None, :]
        correct_token_probs = self.softmax_value[batch_indices, seq_indices, targets]

        # Compute the cross-entropy loss = -log(correct probability)
        loss = -xp.log(correct_token_probs + 1e-9).mean()
        return loss
    

    def backward(self, logits: xp.ndarray, targets: xp.ndarray) -> float:
        """ Compute the gradient of the loss with respect to the predicted values. """

        batch_size, seq_len, _ = logits.shape
        predicted_probs = self._softmax(logits)

        # Create a one-hot vector of the targets for gradient calculation
        real_class = xp.zeros_like(predicted_probs)
        batch_indices = xp.arange(batch_size)[:, None]
        seq_indices = xp.arange(seq_len)[None, :]
        real_class[batch_indices, seq_indices, targets] = 1

        # Compute the gradient of the loss -> partial derivative of the loss with respect to the logits
        grad = (predicted_probs - real_class) / (batch_size * seq_len)
        return grad