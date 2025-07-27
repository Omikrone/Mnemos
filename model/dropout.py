import numpy as np


class Dropout:
    def __init__(self, rate: float):
        self.rate = rate
        self.mask = None

    def forward(self, x: np.ndarray, train: bool = True) -> np.ndarray:
        if train and self.rate > 0.0:
            self.mask = (np.random.rand(*x.shape) > self.rate).astype(np.float32)
            return x * self.mask / (1.0 - self.rate)
        else:
            self.mask = np.ones_like(x)
            return x  # pas de dropout en inference

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        return grad_output * self.mask / (1.0 - self.rate)
    
    def zero_grad(self):
        """ Reset the dropout mask. """
        
        self.mask = None
        
