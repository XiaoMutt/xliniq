import matplotlib.pyplot as plt
from ml.data_set_splitter import DataSetSplitter
import numpy as np

data = DataSetSplitter.get_train_utility_matrix('../output/AllPublicXML.sds').toarray()
m, n = data.shape
fig, (ax0, ax1) = plt.subplots(ncols=2, constrained_layout=True, figsize=(12, 6))

ax0.hist(data.ravel(), bins=100)
ax0.set_yscale('log')
ax0.set_xlabel('TF-IDF')
ax0.set_ylabel('Count')
ax0.set_title('(A) TF-IDF Distribution')

ax1.hist(np.sum(data > 0, axis=0), bins=100)
ax1.set_ylabel('Count')
ax1.set_xlabel('Size of Non-zero TF-IDF')
ax1.set_title('(B) Positive TF-IDF Size Distribution')
# plt.show()
plt.savefig('../results/stats_of_train_data.pdf')
plt.close()