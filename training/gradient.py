import numpy as np


class Param:

    value : int
    gradient : np.ndarray

    def __init__(self, value):
        self.value
        self.gradient = np.zeros_like(value)