from mnemos import xp
from mnemos.transformer.dropout import Dropout
from mnemos.transformer.gradient import Param
from mnemos.transformer.save_model import MLPParams
from mnemos.config.params import DROPOUT_RATE, EMBEDDING_DIM, HIDDEN_DIM


class MLP:
    """ Multi-Layer Perceptron (MLP) class for transforming input embeddings. """

    w_up : Param
    b_up : Param
    h : xp.ndarray
    h_relu : xp.ndarray
    w_down : Param
    b_down : Param
    inputs : xp.ndarray
    dropout : Dropout


    def __init__(self):
        """ Initialize the MLP with random weights and biases. """

        # Random initialization of weights for the first and second layers
        self.w_up = Param(xp.random.randn(EMBEDDING_DIM, HIDDEN_DIM) * 0.01)  # Expanded weight matrix (larger dimension)
        self.w_down = Param(xp.random.randn(HIDDEN_DIM, EMBEDDING_DIM) * 0.01)  # Reduced weight matrix (back to input dimension)

        # Random initialization of biases
        self.b_up = Param(xp.zeros((1, HIDDEN_DIM)))  # Bias for the first layer
        self.b_down = Param(xp.zeros((1, EMBEDDING_DIM)))  # Bias for the second layer
        self.dropout = Dropout(DROPOUT_RATE)  # Dropout layer with a rate of 0.1


    @classmethod
    def from_params(cls, params: MLPParams) -> 'MLP':
        """Create an MLP instance from saved parameters."""

        instance = cls()
        instance.w_up = Param(params.w_up)
        instance.b_up = Param(params.b_up)
        instance.w_down = Param(params.w_down)
        instance.b_down = Param(params.b_down)
        return instance


    def feed_forward(self, inputs : xp.ndarray, train : bool = True) -> xp.ndarray:
        """ Apply the MLP layer to the given inputs. """

        self.inputs = inputs

        # Dot product between the inputs and the weights of the first layer, then add the bias
        self.h = self.inputs @ self.w_up.value + self.b_up.value

        # Apply the ReLU activation function (non-linearity) -> Inactive neurons are set to 0
        self.h_relu = xp.maximum(0, self.h)
        self.h_relu = self.dropout.forward(self.h_relu, train=train)

        # Dot product between the ReLU output and the weights of the second layer, then add the bias
        out = self.h_relu @ self.w_down.value + self.b_down.value

        return out
    

    def backward(self, loss_gradient: xp.ndarray) -> xp.ndarray:
        """ Calcul des gradients du MLP pour la rétropropagation. """

        B, T, _ = loss_gradient.shape

        # === Gradient layer 2 (ReLU -> w_down) ===
        self.w_down.gradient += self.h_relu.reshape(B*T, -1).T @ loss_gradient.reshape(B*T, -1)
        self.b_down.gradient += xp.sum(loss_gradient, axis=(0, 1))

        # === Gradient w.r.t. h_relu ===
        dh_relu = loss_gradient @ self.w_down.value.T

        # === Pass through dropout mask ===
        dh_relu = self.dropout.backward(dh_relu)

        # === Gradient through ReLU ===
        dh_relu = dh_relu * (self.h > 0)

        # === Gradient layer 1 (inputs -> w_up) ===
        self.w_up.gradient += self.inputs.reshape(B*T, -1).T @ dh_relu.reshape(B*T, -1)
        self.b_up.gradient += xp.sum(dh_relu, axis=(0, 1))

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
        self.dropout.zero_grad()


    def get_parameters(self):
        """ Return the MLP parameters for saving. """

        return MLPParams(
            w_up=self.w_up.value,
            b_up=self.b_up.value,
            w_down=self.w_down.value,
            b_down=self.b_down.value
        )