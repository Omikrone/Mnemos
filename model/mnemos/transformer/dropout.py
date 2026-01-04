import numpy as np


class Dropout:
    def __init__(self, rate: float):
        self.rate = rate
        self.masks = []


    def forward(self, x: np.ndarray, train: bool = True) -> np.ndarray:
        """ Apply dropout to the input x during training. """

        if train and self.rate > 0.0:
            mask = (np.random.rand(*x.shape) > self.rate).astype(np.float32)
            self.masks.append(mask)
            return x * mask / (1.0 - self.rate)
        else:
            return x


    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """ Backward pass through the dropout layer. """
        
        mask = self.masks.pop()
        return grad_output * mask / (1.0 - self.rate)
    

    def zero_grad(self):
        """ Reset the dropout mask. """

        self.masks = []
        
