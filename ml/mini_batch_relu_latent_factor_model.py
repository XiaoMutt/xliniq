from base import *
from scipy.sparse import csc_matrix
from ml.abstract_model import AbstractLatentFactorModel
import math


class MiniBatchReLuLatentFactorModel(AbstractLatentFactorModel):
    def __init__(self, tau=-2, k=2, alpha=0.01, lambda_=0.0001, batch_size=10000, max_epoch=1000):
        super(MiniBatchReLuLatentFactorModel, self).__init__(tau, k, alpha, lambda_)
        self._batch_size = batch_size
        self._max_epoch = max_epoch

    def train(self, train_r: csc_matrix):
        def cal_loss():
            rhat = W_theta.dot(R) + b
            rhat[rhat < 0] = 0
            return np.sum(np.power(rhat - R, 2)) / (2 * n)

        R = train_r.toarray()
        m, n = train_r.shape
        theta = np.zeros((m, m))
        b = np.zeros((m, 1))
        W = self._calculate_neighbor_weight_matrix(train_r)

        W_theta = W * theta
        alpha_W = self._alpha * W
        self._training_history = [cal_loss()]
        print(f"Loss before training: {self._training_history[-1]:0.8E}")
        for epoch in range(self._max_epoch):
            pbar = tqdm(range(math.ceil(n / self._batch_size)))
            for batch_index in pbar:
                start = batch_index * self._batch_size
                B = train_r[:, start:start + self._batch_size].toarray()
                bsize = B.shape[1]
                B_hat = W_theta.dot(B) + b
                B_hat_B = (B_hat - B)
                B_hat_B[B_hat < 0] = 0  # multiply ReLu'(r_hat)

                delta_theta = alpha_W / bsize * (B_hat_B.dot(B.transpose())) + self._lambda * theta
                delta_b = self._alpha / bsize * (np.sum(B_hat_B, axis=1, keepdims=True))

                # update theta
                theta = theta - delta_theta
                b = b - delta_b

                # update W_theta
                W_theta = W * theta
                delta_eps = np.sum(np.power(delta_theta, 2) + np.power(delta_b, 2))
                pbar.set_description(f"Epoch {epoch + 1} batch {batch_index + 1} delta={delta_eps:.8E}")
            pbar.close()

            loss = cal_loss()
            self._training_history.append(loss)
            print(f"Final Loss of epoch {epoch + 1}: {loss:0.8E}")

        self._W_theta = W_theta
        self._b = b

    def predict(self, r: np.ndarray):
        res = np.dot(self._W_theta, r) + self._b
        res[res < 0] = 0  # ReLu
        return res
