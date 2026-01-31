from mnemos import xp


class Dropout:
    def __init__(self, rate: float):
        self.rate = rate
        self.masks = []


    def forward(self, x: xp.ndarray, train: bool = True) -> xp.ndarray:
        """ Apply dropout to the input x during training. """

        if train and self.rate > 0.0:
            mask = (xp.random.rand(*x.shape) > self.rate).astype(xp.float32)
            self.masks.append(mask)
            return x * mask / (1.0 - self.rate)
        else:
            return x


    def backward(self, grad_output: xp.ndarray) -> xp.ndarray:
        """ Backward pass through the dropout layer. """
        
        mask = self.masks.pop()
        return grad_output * mask / (1.0 - self.rate)
    

    def zero_grad(self):
        """ Reset the dropout mask. """

        self.masks = []
        
