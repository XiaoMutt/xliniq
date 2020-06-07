import matplotlib.pyplot as plt
import numpy as np
import pickle
import pandas as pd
import seaborn as sns
import scipy.special as sps


def integral(tau, k):
    k_ = 1 / k
    ginc = sps.gammaincc(k_, 0) - sps.gammaincc(k_, -tau)
    res = ginc / (k * (-1) ** k_ * tau ** k_) * sps.gamma(k_)
    return np.absolute(res)


taus = [-10, -1, -0.1]
ks = [2, 4, 16]
lambda_s = [1, 1e-1, 1e-2, 1e-4, 1e-6]
paramses = []
tmp = {
    'name': [],
    'tau': [],
    'k': [],
    'area': [],
    'lambda': [],
    'mser': [],
    'aptk': []
}
for tau in taus:
    for k in ks:

        for lam in lambda_s:
            params = {
                'tau': tau, 'k': k, 'lambda_': lam, 'alpha': 0.1, 'max_epoch': 30,
            }
            paramses.append(params)

            save_name = f"model{'_'.join(map(str, params.values()))}"

            with open(f'../output/{save_name}.pickle', 'rb') as f:
                model = pickle.load(f)

            tmp['name'].append(save_name)
            tmp['tau'].append(tau)
            tmp['k'].append(k)
            tmp['area'].append(integral(tau, k))
            tmp['lambda'].append(lam)
            tmp['mser'].append(model.mse_test[1])
            tmp['aptk'].append(model.apk_test[0])

df = pd.DataFrame(tmp)
plt.figure(figsize=(16, 12))
sns.pairplot(df, hue='lambda', vars=['tau', 'k', 'area', 'aptk', 'mser'], diag_kind='hist')
plt.show()
plt.savefig('../results/finding_params.pdf')
