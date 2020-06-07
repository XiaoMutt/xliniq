from base import *
from ml.abstract_model import AbstractLatentFactorModel
from ml.abstract_tester import AbstractTester
from scipy.sparse import csc_matrix


class MseTester(AbstractTester):
    """
    Test the model's using Mean Squared Error (MSE). Concretely, this works by the following steps:
    """

    def test(self, r):
        masks = self.choose_mask_indices(r)
        new_r = np.array(r) * masks

        predicted_r = self._model.predict(new_r)

        relevant = (predicted_r > 0) | (r > 0)

        return (np.power(predicted_r - r, 2).ravel(),
                np.power(predicted_r[relevant] - r[relevant], 2).ravel())

    def __call__(self, docs: csc_matrix) -> tp.Tuple:
        """
        Randomly mask the non-zero values in the docs, and return the mean squared error of the prediction
        :param docs: 2-d array. each column vector is a mesh index vector
        :return: mean squared error, mean squared error of relevant
        """
        mse = []
        mser = []

        for col in tqdm(range(docs.shape[1])):
            r = docs[:, col].toarray()
            if np.sum(r) > 0:
                a, b = self.test(r)
                mse.extend(a)
                mser.extend(b)
        res = (np.mean(mse), np.mean(mser))
        self._model.mse_test = res
        return res
