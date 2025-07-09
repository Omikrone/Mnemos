import numpy as np


class Linear:
    def __init__(self, in_dim, out_dim):
        self.value = np.random.randn(in_dim, out_dim) * 0.01
        self.bias = np.zeros((1, out_dim))

        self.gradient = np.zeros_like(self.value)
        self.bias_gradient = np.zeros_like(self.bias)

    def forward(self, x):
        self.input = x
        return x @ self.value + self.bias

    def backward(self, grad_output):
        B, T, _ = grad_output.shape
        self.gradient += np.einsum('bti,btj->ij', self.input, grad_output) / (B * T)
        self.bias_gradient += np.sum(grad_output, axis=(0, 1), keepdims=True) / (B * T)
        grad_input = grad_output @ self.value.T
        return grad_input
