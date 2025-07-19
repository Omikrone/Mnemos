import numpy as np

from model.gradient import Param
from model.embeddings import EMBEDDING_DIMENSION
from model.save_model import MLPParams

HIDDEN_DIMENSION = 128  # Hidden dimension for the MLP


class MLP:
    """ Multi-Layer Perceptron (MLP) class for transforming input embeddings. """

    w_up : Param
    b_up : Param
    h : np.ndarray
    h_relu : np.ndarray
    w_down : Param
    b_down : Param
    inputs : np.ndarray


    def __init__(self):
        """ Initialize the MLP with random weights and biases. """

        # Random initialization of weights for the first and second layers
        self.w_up = Param(np.random.randn(EMBEDDING_DIMENSION, HIDDEN_DIMENSION) * 0.01)  # Expanded weight matrix (larger dimension)
        self.w_down = Param(np.random.randn(HIDDEN_DIMENSION, EMBEDDING_DIMENSION) * 0.01)  # Reduced weight matrix (back to input dimension)

        # Random initialization of biases
        self.b_up = Param(np.zeros((1, HIDDEN_DIMENSION)))  # Bias for the first layer
        self.b_down = Param(np.zeros((1, EMBEDDING_DIMENSION)))  # Bias for the second layer


    @classmethod
    def from_params(cls, params: MLPParams) -> 'MLP':
        """Create an MLP instance from saved parameters."""

        instance = cls()
        instance.w_up = Param(params.w_up)
        instance.b_up = Param(params.b_up)
        instance.w_down = Param(params.w_down)
        instance.b_down = Param(params.b_down)
        return instance


    def feed_forward(self, inputs : np.ndarray) -> np.ndarray:
        """ Apply the MLP layer to the given inputs. """

        self.inputs = inputs

        # Dot product between the inputs and the weights of the first layer, then add the bias
        self.h = self.inputs @ self.w_up.value + self.b_up.value

        # Apply the ReLU activation function (non-linearity) -> Inactive neurons are set to 0
        self.h_relu = np.maximum(0, self.h)

        # Dot product between the ReLU output and the weights of the second layer, then add the bias
        out = self.h_relu @ self.w_down.value + self.b_down.value

        return out
    

    def backward(self, loss_gradient: np.ndarray) -> np.ndarray:
        """ Calcul des gradients du MLP pour la rétropropagation. """

        B, T, _ = loss_gradient.shape

        # Reshape matrices to 2D for matrix multiplications, then compute the gradient for the second layer
        # from the loss gradient
        self.w_down.gradient += self.h_relu.reshape(B*T, -1).T @ loss_gradient.reshape(B*T, -1)

        # Compute the gradient of the bias for the second layer
        self.b_down.gradient += np.sum(loss_gradient, axis=(0, 1))

        # Compute the gradient of the ReLU
        dh_relu = loss_gradient @ self.w_down.value.T
        dh_relu = dh_relu * (self.h > 0) # Derivative of ReLU, 0 if h <= 0, else 1

        # Compute the gradient for the first layer (same principle as for the second)
        self.w_up.gradient += self.inputs.reshape(B*T, -1).T @ dh_relu.reshape(B*T, -1)

        # Compute the gradient of the bias for the first layer
        self.b_up.gradient += np.sum(dh_relu, axis=(0, 1))

        # Compute the gradient of the inputs for the previous layer
        dx = dh_relu @ self.w_up.value.T

        return dx
    

    def step(self, lr : float):
        """ Update the MLP weights and biases based on the computed gradients. """

        self.w_up.step(lr)
        self.w_down.step(lr)
        self.b_up.step(lr)
        self.b_down.step(lr)


    def zero_grad(self):
        """ Reset the gradients of the MLP weights and biases. """

        self.w_up.zero_grad()
        self.w_down.zero_grad()
        self.b_up.zero_grad()
        self.b_down.zero_grad()


    def get_parameters(self):
        """ Return the MLP parameters for saving. """

        return MLPParams(
            w_up=self.w_up.value,
            b_up=self.b_up.value,
            w_down=self.w_down.value,
            b_down=self.b_down.value
        )