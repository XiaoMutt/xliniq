from base import *
from abc import ABC, abstractmethod
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import norm


class AbstractLatentFactorModel(ABC):
    def __init__(self, tau, k, alpha, lambda_):
        self._tau = tau
        self._k = k
        self._alpha = alpha
        self._lambda = lambda_
        self._b = None
        self._W_theta = None
        self._training_history = None
        self.mse_test = None
        self.apk_test = None

    def _calculate_neighbor_weight_matrix(self, train_r: csc_matrix) -> np.ndarray:
        l2norm = norm(train_r, ord=2, axis=1)
        l2norm[l2norm == 0] = 1  # handel 0 vector
        U: csc_matrix = train_r.multiply(1 / l2norm.reshape(-1, 1))
        UUT = U.dot(U.transpose()).toarray()  # dense
        W = np.exp(self._tau * np.power(1 - UUT, self._k))
        np.fill_diagonal(W, 0)
        return W

    @abstractmethod
    def train(self, *args, **kwargs):
        raise Exception("Not implemented yet.")

    @abstractmethod
    def predict(self, arg):
        raise Exception("Not implemented yet.")
