import numpy as np


class Param:
    """ Class representing a parameter with its value and gradient. """

    value : np.ndarray
    gradient : np.ndarray

    def __init__(self, value):
        """ Initialize the parameter with a value and zero gradient. """

        self.value = value
        self.gradient = np.zeros_like(value)

    def step(self, lr : float):
        """ Update the weight with gradient descent """
        
        self.value -= lr * self.gradient

    def zero_grad(self):
        """ Reset the gradient """
        
        self.gradient.fill(0)