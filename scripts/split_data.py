from ml.data_set_splitter import DataSetSplitter

if __name__ == '__main__':
    import pickle

    dataset = 'AllPublicXML'
    with open(f'../output/{dataset}.ctdmc', 'rb') as f:
        ctdmc = pickle.load(f)

    path = DataSetSplitter.dump_split_data_set(ctdmc, f'../output/full{dataset}.sds', frequency_threshold=0)
