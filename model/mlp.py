import numpy as np

from model.gradient import Param
from model.embeddings import EMBEDDING_DIMENSION
from model.save_model import MLPParams

HIDDEN_DIMENSION = 128  # Dimension cachée pour le MLP


class MLP:

    w_up : Param
    b_up : Param
    h : np.ndarray
    h_relu : np.ndarray
    w_down : Param
    b_down : Param
    inputs : np.ndarray


    def __init__(self):

        # Initialisation aléatoire des poids de la première et seconde couche
        self.w_up = Param(np.random.randn(EMBEDDING_DIMENSION, HIDDEN_DIMENSION) * 0.01)  # Matrice de poids aggrandie (de plus grande dimension)
        self.w_down = Param(np.random.randn(HIDDEN_DIMENSION, EMBEDDING_DIMENSION) * 0.01)  # Matrice de poids réduite (retour à la dimension d'entrée)

        # Initialisation à 0 des biais
        self.b_up = Param(np.zeros((1, HIDDEN_DIMENSION)))  # Biais de la première couche
        self.b_down = Param(np.zeros((1, EMBEDDING_DIMENSION)))  # Biais de la seconde couche


    @classmethod
    def from_params(cls, params: MLPParams) -> 'MLP':
        """Crée une instance de MLP à partir des paramètres sauvegardés."""

        instance = cls()
        instance.w_up = Param(params.w_up)
        instance.b_up = Param(params.b_up)
        instance.w_down = Param(params.w_down)
        instance.b_down = Param(params.b_down)
        return instance


    def feed_forward(self, inputs : np.ndarray) -> np.ndarray:
        """ Applique la couche MLP sur les entrées données. """

        self.inputs = inputs

        # Produit scalaire entre les entrées et les poids de la première couche, puis ajout du biais
        self.h = self.inputs @ self.w_up.value + self.b_up.value

        # Application de la fonction d'activation ReLU (non-linéarité) -> Neurones inactifs sont mis à 0
        self.h_relu = np.maximum(0, self.h)

        # Produit scalaire entre la sortie de la ReLU et les poids de la seconde couche, puis ajout du biais
        out = self.h_relu @ self.w_down.value + self.b_down.value

        return out
    

    def backward(self, loss_gradient: np.ndarray) -> np.ndarray:
        """ Calcul des gradients du MLP pour la rétropropagation. """

        B, T, _ = loss_gradient.shape

        # Redimensionnement des matrices en 2D pour les multiplications matricielles, puis calcul ddu gradient de la seconde couche
        # à partir du gradient de la perte
        self.w_down.gradient += self.h_relu.reshape(B*T, -1).T @ loss_gradient.reshape(B*T, -1)

        # Calcul du gradient du biais de la seconde couche
        self.b_down.gradient += np.sum(loss_gradient, axis=(0, 1))

         # Calcul du gradient de la ReLU
        dh_relu = loss_gradient @ self.w_down.value.T
        dh_relu = dh_relu * (self.h > 0) # Dérivée de ReLU, 0 si h <= 0, sinon 1

        # Calcul du gradient de la première couche (même principe que pour la seconde)
        self.w_up.gradient += self.inputs.reshape(B*T, -1).T @ dh_relu.reshape(B*T, -1)

        # Calcul du gradient du biais de la première couche
        self.b_up.gradient += np.sum(dh_relu, axis=(0, 1))

        # Calcul du gradient des entrées pour la couche précédente
        dx = dh_relu @ self.w_up.value.T

        return dx
    

    def step(self, lr : float):
        """ Met à jour les poids et biais du MLP en fonction des gradients calculés. """

        self.w_up.step(lr)
        self.w_down.step(lr)
        self.b_up.step(lr)
        self.b_down.step(lr)


    def zero_grad(self):
        """ Réinitialise les gradients des poids et biais du MLP. """

        self.w_up.zero_grad()
        self.w_down.zero_grad()
        self.b_up.zero_grad()
        self.b_down.zero_grad()


    def get_parameters(self):
        """ Retourne les paramètres du MLP pour la sauvegarde. """
        
        return MLPParams(
            w_up=self.w_up.value,
            b_up=self.b_up.value,
            w_down=self.w_down.value,
            b_down=self.b_down.value
        )