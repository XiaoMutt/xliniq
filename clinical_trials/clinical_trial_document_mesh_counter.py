from mesh.trie.mesh_trie import MeshTrie
from ir.string_indexer import StringIndexer
from clinical_trials.clinical_trial_document import ClinicalTrialDocument
from collections import Counter
from math import log10


class ClinicalTrialDocumentMeshCounter(object):
    def __init__(self, mesh_trie: MeshTrie):
        self._mesh_trie = mesh_trie
        self._doc_indexer = StringIndexer()

        # array of mesh_index->counter. Counter is doc_index->count (a float number)
        self._counter = [None] * mesh_trie.total_meshes

    def process(self, record: ClinicalTrialDocument):
        doc_index = self._doc_indexer.add(record.nct_id)
        for attr in ClinicalTrialDocument.MESH_ATTRIBUTES:
            # count mesh indices in the record attributes
            counter = self._mesh_trie.count_mesh_indices(getattr(record, attr))
            # add to count
            for mesh_index, count in counter.items():
                if self._counter[mesh_index] is None:
                    self._counter[mesh_index] = Counter()
                self._counter[mesh_index][doc_index] += count

    def tf(self, mesh_index: int, doc_index: int):
        return self._counter[mesh_index][doc_index]

    def idf(self, mesh_index: int):
        if self._counter[mesh_index] is None:
            return len(self._doc_indexer)
        else:
            return len(self._doc_indexer) / len(self._counter[mesh_index])

    def tf_idf(self, mesh_index: int, doc_index: int) -> float:
        """
        Return TF-IDF, which is >=0. It is 0 when term not showing up the document.
        :param mesh_index:
        :param doc_index:
        :return:
        """
        return log10(1 + self.tf(mesh_index, doc_index)) * log10(self.idf(mesh_index))

    def __iter__(self):
        return self._counter.__iter__()

    def num_processed_docs(self):
        return len(self._doc_indexer)
