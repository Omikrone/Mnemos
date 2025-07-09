import numpy as np


class Param:

    value : np.ndarray
    gradient : np.ndarray

    def __init__(self, value):
        self.value = value
        self.gradient = np.zeros_like(value)

    def step(self, lr : float):
        """ Mise à jour du poids avec la descente de gradient """
        
        self.value -= lr * self.gradient

    def zero_grad(self):
        """ Réinitialisation du gradient """
        
        self.gradient.fill(0)