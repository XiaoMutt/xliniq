from ml.mse_tester import MseTester
from ml.average_precision_k_tester import AveragePrecisionKTester
from ml.data_set_splitter import DataSetSplitter
import pickle

with open('/output/model-0.001_1000.0_0.01_0.1_60.pickle', 'rb') as f:
    model = pickle.load(f)

print(model.mse_test)
print(model.apk_test)

TEST = DataSetSplitter.get_test_utility_matrix('../output/AllPublicXML.sds')

mset = MseTester(model)
mse = mset(TEST)
print(mse)
apkt = AveragePrecisionKTester(model)
apk = apkt(TEST)
print(apk)
