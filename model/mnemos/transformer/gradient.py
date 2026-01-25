from mnemos import xp


class Param:
    """ Class representing a parameter with its value and gradient. """

    value : xp.ndarray
    gradient : xp.ndarray

    def __init__(self, value):
        """ Initialize the parameter with a value and zero gradient. """

        self.value = value
        self.gradient = xp.zeros_like(value)

    def step(self, lr : float):
        """ Update the weight with gradient descent """
        
        self.value -= lr * self.gradient

    def zero_grad(self):
        """ Reset the gradient """
        
        self.gradient.fill(0)