import numpy as np

from model.gradient import Param
from model.embeddings import EMBEDDING_DIMENSION
from model.save_model import MLPParams

HIDDEN_DIMENSION = 128  # Dimension cachée pour le MLP


class MLP:

    w1 : Param
    b1 : Param
    h : np.ndarray
    h_relu : np.ndarray
    w2 : Param
    b2 : Param
    inputs : np.ndarray

    def __init__(self):

        # Initialisation aléatoire des poids de la première et seconde couche
        self.w1 = Param(np.random.randn(EMBEDDING_DIMENSION, HIDDEN_DIMENSION) * 0.01)  # Poids de la première couche
        self.w2 = Param(np.random.randn(HIDDEN_DIMENSION, EMBEDDING_DIMENSION) * 0.01)  # Poids de la seconde couche

        # Initialisation à 0 des biais
        self.b1 = Param(np.zeros((1, HIDDEN_DIMENSION)))  # Biais de la première couche
        self.b2 = Param(np.zeros((1, EMBEDDING_DIMENSION)))  # Biais de la seconde couche

    @classmethod
    def from_params(cls, params: MLPParams):
        instance = cls()
        instance.w1 = Param(params.w1)
        instance.b1 = Param(params.b1)
        instance.w2 = Param(params.w2)
        instance.b2 = Param(params.b2)
        return instance

    def feed_forward(self, inputs : np.ndarray) -> np.ndarray:
        self.inputs = inputs
        self.h = self.inputs @ self.w1.value + self.b1.value
        self.h_relu = np.maximum(0, self.h)
        out = self.h_relu @ self.w2.value + self.b2.value

        return out
    
    def backward(self, loss_gradient: np.ndarray) -> np.ndarray:
        B, T, _ = loss_gradient.shape

        # Backprop couche 2
        self.w2.gradient += self.h_relu.reshape(B*T, -1).T @ loss_gradient.reshape(B*T, -1)
        self.b2.gradient += np.sum(loss_gradient, axis=(0, 1))

        # Backprop ReLU
        dh_relu = loss_gradient @ self.w2.value.T
        dh_relu = dh_relu * (self.h > 0)  # dérivée ReLU

        # Backprop couche 1
        self.w1.gradient += self.inputs.reshape(B*T, -1).T @ dh_relu.reshape(B*T, -1)
        self.b1.gradient += np.sum(dh_relu, axis=(0, 1))

        dx = dh_relu @ self.w1.value.T
        return dx
    
    def step(self, lr : float):
        self.w1.step(lr)
        self.w2.step(lr)
        self.b1.step(lr)
        self.b2.step(lr)

    def zero_grad(self):
        self.w1.zero_grad()
        self.w2.zero_grad()
        self.b1.zero_grad()
        self.b2.zero_grad()

    def get_parameters(self):
        return MLPParams(
            w1=self.w1.value,
            b1=self.b1.value,
            w2=self.w2.value,
            b2=self.b2.value
        )