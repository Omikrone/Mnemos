import numpy as np

from mnemos.transformer.gradient import Param
from mnemos.transformer.save_model import LayerNormParams
from mnemos.config.params import EPS


class LayerNorm:
    """ Layer Normalization class for normalizing inputs to avoid exploding or vanishing gradients. """

    gamma : Param
    beta : Param
    eps : float
    x : np.ndarray
    mean : np.ndarray
    var : np.ndarray
    std : np.ndarray
    dim : int
    x_norm : np.ndarray


    def __init__(self, dim, eps=EPS):

        # Initialisation of the parameters gamma and beta
        self.gamma = Param(np.ones((1, 1, dim))) # Weights for normalization
        self.beta = Param(np.zeros((1, 1, dim))) # Biases for normalization

        # Constant to avoid division by zero
        self.eps = eps


    @classmethod
    def from_params(cls, params: LayerNormParams) -> 'LayerNorm':
        """ Create a LayerNorm instance from saved parameters. """

        instance = cls(dim=params.gamma.shape[-1], eps=params.eps)
        instance.gamma = Param(params.gamma)
        instance.beta = Param(params.beta)
        return instance


    def forward(self, x : np.ndarray) -> np.ndarray:
        """ Normalize the inputs x to avoid exploding or vanishing gradients. """

        self.x = x

        # Compute the mean, variance, and standard deviation for normalization
        self.mean = np.mean(x, axis=-1, keepdims=True)
        self.var = np.var(x, axis=-1, keepdims=True)
        self.std = np.maximum(np.sqrt(self.var + self.eps), 1e-6)

        # Normalize the inputs
        self.x_norm = (x - self.mean) / self.std

        # Apply normalization with learned parameters gamma and beta
        result = self.gamma.value * self.x_norm + self.beta.value

        return result


    def backward(self, dout):
        """ Compute the gradients of the Layer Norm for backpropagation. """
        
        _, _, D = self.x.shape

        # Compute gradients for parameters gamma and beta
        self.gamma.gradient = np.sum(dout * self.x_norm, axis=(0, 1), keepdims=True)
        self.beta.gradient = np.sum(dout, axis=(0, 1), keepdims=True)

        # Compute gradient of normalization, variance, and mean
        dx_norm = dout * self.gamma.value
        dvar = np.sum(dx_norm * (self.x - self.mean) * -0.5 * self.std**-3, axis=-1, keepdims=True)
        dmean = np.sum(-dx_norm / self.std, axis=-1, keepdims=True) + dvar * np.mean(-2.0 * (self.x - self.mean), axis=-1, keepdims=True)
        dx = dx_norm / self.std + dvar * 2.0 * (self.x - self.mean) / D + dmean / D

        # Compute gradient of inputs
        dx = dx_norm / self.std + dvar * 2.0 * (self.x - self.mean) / D + dmean / D
        
        return dx
    

    def step(self, lr):
        """ Update the parameters gamma and beta with gradient descent. """

        self.gamma.step(lr)
        self.beta.step(lr)


    def zero_grad(self):
        """ Reset the gradients of the parameters gamma and beta. """

        self.gamma.zero_grad()
        self.beta.zero_grad()


    def get_params(self):
        """ Return the parameters gamma and beta for saving. """

        return LayerNormParams(
            gamma=self.gamma.value,
            beta=self.beta.value,
            eps=self.eps
        )