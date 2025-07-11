import numpy as np
from model.gradient import Param
from model.save_model import LayerNormParams


class LayerNorm:

    gamma : Param
    beta : Param
    eps : float
    x : np.ndarray
    mean : np.ndarray
    var : np.ndarray
    std : np.ndarray
    dim : int
    x_norm : np.ndarray

    def __init__(self, dim, eps=1e-5):
        self.gamma = Param(np.ones((1, 1, dim)))
        self.beta = Param(np.zeros((1, 1, dim)))
        self.eps = eps

    @classmethod
    def from_params(cls, params: LayerNormParams) -> 'LayerNorm':
        instance = cls(dim=params.gamma.shape[-1], eps=params.eps)
        instance.gamma = Param(params.gamma)
        instance.beta = Param(params.beta)
        return instance

    def forward(self, x):
        self.x = x

        self.mean = np.mean(x, axis=-1, keepdims=True)
        self.var = np.var(x, axis=-1, keepdims=True)
        self.std = np.sqrt(self.var + self.eps)
        self.x_norm = (x - self.mean) / self.std
        return self.gamma.value * self.x_norm + self.beta.value

    def backward(self, dout):
        N, T, D = self.x.shape

        # Gradients wrt gamma & beta (paramètres appris)
        self.gamma.gradient = np.sum(dout * self.x_norm, axis=(0, 1), keepdims=True)
        self.beta.gradient = np.sum(dout, axis=(0, 1), keepdims=True)

        dx_norm = dout * self.gamma.value

        dvar = np.sum(dx_norm * (self.x - self.mean) * -0.5 * self.std**-3, axis=-1, keepdims=True)
        dmean = np.sum(-dx_norm / self.std, axis=-1, keepdims=True) + dvar * np.mean(-2.0 * (self.x - self.mean), axis=-1, keepdims=True)

        dx = dx_norm / self.std + dvar * 2.0 * (self.x - self.mean) / D + dmean / D
        return dx
    
    def step(self, lr):
        self.gamma.step(lr)
        self.beta.step(lr)

    def zero_grad(self):
        self.gamma.zero_grad()
        self.beta.zero_grad()

    def get_params(self):
        return LayerNormParams(
            gamma=self.gamma.value,
            beta=self.beta.value,
            eps=self.eps
        )