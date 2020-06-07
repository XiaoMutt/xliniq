from base import *
from scipy.sparse import lil_matrix, csc_matrix
from clinical_trials.clinical_trial_document_mesh_counter import ClinicalTrialDocumentMeshCounter


class DATA_SET_INDICATOR(object):
    TRAIN = 0
    VALIDATE = 1
    TEST = 2


class DataSetSplitter(object):
    __FILE_EXTENSION__ = ".sds"  # split data set

    class SparseMatrixCompressor(object):
        @staticmethod
        def compress(src: tp.List) -> bytes:
            # |4bytes total, 4bytes total_non_zeros|non zero float32 array|non zero int32 index|
            arr = np.array(src, dtype='f4')
            total = np.uint32(len(arr))
            non_zeros_num = np.uint32(np.count_nonzero(arr))
            non_zeros_values = arr[arr != 0]
            non_zeros_indices = np.argwhere(arr).ravel().astype('u4')
            return total.tobytes() + non_zeros_num.tobytes() + non_zeros_indices.tobytes() + non_zeros_values.tobytes()

        @staticmethod
        def decompress(reader: tp.BinaryIO):
            total, non_zeros_num = np.frombuffer(reader.read(4 + 4), dtype='u4')
            non_zeros_indices = np.frombuffer(reader.read(4 * non_zeros_num), 'u4')
            non_zeros_values = np.frombuffer(reader.read(4 * non_zeros_num), 'f4')
            res = np.zeros(total, dtype='f4')
            np.put(res, non_zeros_indices, non_zeros_values)
            res[non_zeros_indices] = non_zeros_values
            return res

    @classmethod
    def dump_split_data_set(cls, ctdmc: ClinicalTrialDocumentMeshCounter, output_file_path: str,
                            split_fractions: tp.Iterable = (0.8, 0.1, 0.1),
                            frequency_threshold: tp.Union[int, float] = 0.005):
        """

        :param ctdmc:
        :param output_file_path:
        :param split_fractions: the fractions size of the train, validate, train samples.
        :param frequency_threshold: the mesh_indices showing up in the number of documents less than the threshold will
         not be kept. If the frequency_threshold is a float then frequency_threshold*num_processed_docs is the
         threshold
        :return:
        """

        # get rid of low frequent mesh_indices, so need to remap the indices
        if type(frequency_threshold) is float:
            frequency_threshold = frequency_threshold * ctdmc.num_processed_docs()
        mesh_index_map = []  # kept mesh index -> ctdmc mesh index
        for mesh_index, counter in tqdm(enumerate(ctdmc)):
            if counter is not None and len(counter) >= frequency_threshold:
                mesh_index_map.append(mesh_index)

        num_mesh = len(mesh_index_map)
        num_docs = ctdmc.num_processed_docs()
        split_choice_map = np.random.choice((DATA_SET_INDICATOR.TRAIN,
                                             DATA_SET_INDICATOR.VALIDATE,
                                             DATA_SET_INDICATOR.TEST), size=num_docs,
                                            p=np.array(split_fractions) / np.sum(split_fractions))
        with open(output_file_path, 'wb') as writer:
            # first 8 bytes are
            writer.write(np.array([num_mesh, num_docs], dtype='u4').tobytes())
            # mesh_index_map
            writer.write(np.array(mesh_index_map, dtype='u4').tobytes())
            # split_choice_map
            writer.write(split_choice_map.astype('b').tobytes())
            # utility_matrix
            for mesh_index in tqdm(mesh_index_map):
                writer.write(cls.SparseMatrixCompressor.compress(
                    [ctdmc.tf_idf(mesh_index, doc_index) for doc_index in range(num_docs)]))

        # return the two map for testing purpose
        return mesh_index_map, split_choice_map

    @classmethod
    def get_utility_matrix(cls, file_path: str, data_set_indicator: int) -> csc_matrix:
        with open(file_path, 'rb') as reader:
            num_mesh, num_docs = np.frombuffer(reader.read(4 + 4), dtype='u4')
            reader.seek(4 * num_mesh, io.SEEK_CUR)  # jump over mesh_index_map
            split_choice_map = np.frombuffer(reader.read(num_docs), dtype='b')
            chosen_docs = split_choice_map == data_set_indicator
            num_chosen_docs = np.sum(chosen_docs)
            res = lil_matrix((num_mesh, num_chosen_docs))
            for i in tqdm(range(num_mesh)):
                row = cls.SparseMatrixCompressor.decompress(reader)
                res[i] = row[chosen_docs]  # set row to the chosen docs' values
        return res.tocsc()

    @classmethod
    def get_train_utility_matrix(cls, file_path: str) -> csc_matrix:
        return cls.get_utility_matrix(file_path, DATA_SET_INDICATOR.TRAIN)

    @classmethod
    def get_validate_utility_matrix(cls, file_path: str) -> csc_matrix:
        return cls.get_utility_matrix(file_path, DATA_SET_INDICATOR.VALIDATE)

    @classmethod
    def get_test_utility_matrix(cls, file_path: str) -> csc_matrix:
        return cls.get_utility_matrix(file_path, DATA_SET_INDICATOR.TEST)

    @classmethod
    def get_mesh_index_map(cls, file_path: str) -> np.ndarray:
        with open(file_path, 'rb') as reader:
            num_mesh, num_docs = np.frombuffer(reader.read(4 + 4), dtype='u4')
            res = np.frombuffer(reader.read(num_mesh * 4), dtype='u4')
        return res

    @classmethod
    def get_split_choice_map(cls, file_path: str) -> np.ndarray:
        with open(file_path, 'rb') as reader:
            num_mesh, num_docs = np.frombuffer(reader.read(4 + 4), dtype='u4')
            reader.seek(num_mesh * 4, io.SEEK_CUR)
            res = np.frombuffer(reader.read(num_docs), dtype='b')
        return res

