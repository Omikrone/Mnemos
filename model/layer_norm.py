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

        # Initialisation des paramètres gamma et beta pour la normalisation
        self.gamma = Param(np.ones((1, 1, dim))) # Poids pour la normalisation
        self.beta = Param(np.zeros((1, 1, dim))) # Biais pour la normalisation

        # Constante pour éviter la division par zéro
        self.eps = eps


    @classmethod
    def from_params(cls, params: LayerNormParams) -> 'LayerNorm':
        """ Crée une instance de LayerNorm à partir des paramètres sauvegardés. """

        instance = cls(dim=params.gamma.shape[-1], eps=params.eps)
        instance.gamma = Param(params.gamma)
        instance.beta = Param(params.beta)
        return instance


    def forward(self, x : np.ndarray) -> np.ndarray:
        """ Normalise les entrées x pour éviter les problèmes d'explosion ou de disparition du gradient. """

        self.x = x

        # Calcul de la moyenne, de la variance et de l'écart type pour la normalisation
        self.mean = np.mean(x, axis=-1, keepdims=True)
        self.var = np.var(x, axis=-1, keepdims=True)
        self.std = np.sqrt(self.var + self.eps)

        # Normalisation des entrées
        self.x_norm = (x - self.mean) / self.std

        # Application de la normalisation avec les paramètres gamma et beta (appris)
        result = self.gamma.value * self.x_norm + self.beta.value

        return result


    def backward(self, dout):
        """ Calcul des gradients de la Layer Norm pour la rétropropagation. """
        _, _, D = self.x.shape

        # Calcul des gradients pour les paramètres gamma et beta
        self.gamma.gradient = np.sum(dout * self.x_norm, axis=(0, 1), keepdims=True)
        self.beta.gradient = np.sum(dout, axis=(0, 1), keepdims=True)

        # Calcul du gradient de la normalisation, de la variance puis de la moyenne
        dx_norm = dout * self.gamma.value
        dvar = np.sum(dx_norm * (self.x - self.mean) * -0.5 * self.std**-3, axis=-1, keepdims=True)
        dmean = np.sum(-dx_norm / self.std, axis=-1, keepdims=True) + dvar * np.mean(-2.0 * (self.x - self.mean), axis=-1, keepdims=True)

        # Calcul du gradient des entrées
        dx = dx_norm / self.std + dvar * 2.0 * (self.x - self.mean) / D + dmean / D
        
        return dx
    

    def step(self, lr):
        """ Met à jour les paramètres gamma et beta avec la descente de gradient. """

        self.gamma.step(lr)
        self.beta.step(lr)


    def zero_grad(self):
        """ Réinitialise les gradients des paramètres gamma et beta. """

        self.gamma.zero_grad()
        self.beta.zero_grad()


    def get_params(self):
        """ Retourne les paramètres gamma et beta pour la sauvegarde. """

        return LayerNormParams(
            gamma=self.gamma.value,
            beta=self.beta.value,
            eps=self.eps
        )