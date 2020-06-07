from base import *
from ml.abstract_model import AbstractLatentFactorModel
from ml.abstract_tester import AbstractTester
from scipy.sparse import csc_matrix
import heapq


class AveragePrecisionKTester(AbstractTester):
    """
        Test the model using the precision of topK. Concretely, by the following steps:
        (1) a fraction of mesh terms (fraction_of_masking) having non zero tf-idf will be masked to zero
        (2) the masked record will undergo prediction by the given model
        (3) the top K predictions (previously zero, but now has a non zero tf-idf) will be used to calculate precision.
        NOTE: rank by the tf-idf is not considered
    """

    def __init__(self, model: AbstractLatentFactorModel, k=3, fraction_of_masking=0.8):
        super(AveragePrecisionKTester, self).__init__(model, fraction_of_masking)
        self._k = k

    def test(self, r: np.ndarray) -> tp.Tuple[float, float]:
        """
        return Precision@K
        :param r: test record
        :return: precision @ K
        """
        masks = self.choose_mask_indices(r)
        new_r = r * masks

        zeros_before_prediction = new_r == 0
        predicted_r = self._model.predict(new_r)
        positive_after_prediction = predicted_r > 0
        index_tf_idf = [(idx, value)
                        for idx, value in enumerate(predicted_r.ravel())
                        if zeros_before_prediction[idx] and positive_after_prediction[idx]]

        predicted = heapq.nlargest(self._k, index_tf_idf, key=lambda x: x[1])
        if len(predicted):
            random_picked = np.random.choice(np.argwhere(zeros_before_prediction.ravel() == True).ravel(),
                                             len(predicted))
            return (sum([r[idx][0] != 0 for idx, _ in predicted]) / len(predicted),
                    sum([r[idx][0] != 0 for idx in random_picked]) / len(random_picked))
        else:
            return 0, 0

    def __call__(self, docs: csc_matrix)->tp.Tuple:
        """
        Randomly mask the non-zero values in the docs, and return the mean squared error of the prediction
        :param docs: 2-d array. each column vector is a mesh index vector
        :return: average precision at K of the model, average precision at K of random guess
        """
        model = []
        rand = []
        for col in tqdm(range(docs.shape[1])):
            r = docs[:, col].toarray()
            if np.sum(r) > 0:
                m, r = self.test(r)
                model.append(m)
                rand.append(r)
        res = np.mean(model), np.mean(rand)
        self._model.apk_test = res
        return res
