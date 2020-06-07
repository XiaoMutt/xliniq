from abc import ABC, abstractmethod
import numpy as np
from ml.abstract_model import AbstractLatentFactorModel
from scipy.sparse import csc_matrix


class AbstractTester(ABC):
    def __init__(self, model: AbstractLatentFactorModel, fraction_of_masking=0.5):
        """
        Test the model
        :param model: Model to be tested
        :param fraction_of_masking: the fraction of the non-zero values that will be masked to zero
        """
        self._model = model
        self._fraction_of_masking = fraction_of_masking

    def choose_mask_indices(self, r: np.ndarray) -> np.ndarray:
        """

        :param r: mx1 np array to be masked
        :return:
        """
        non_zeros_indices = np.argwhere(r.ravel() != 0).ravel()
        indices = np.random.choice(non_zeros_indices, int(len(non_zeros_indices) * self._fraction_of_masking))
        res = np.ones(r.shape)
        for idx in indices:
            res[idx][0] = 0
        return res

    @abstractmethod
    def test(self, r: np.ndarray):
        pass

    @abstractmethod
    def __call__(self, docs: csc_matrix):
        pass
