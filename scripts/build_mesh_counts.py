from clinical_trials.clinical_trial_document_xml_zip_file_reader import ClinicalTrialDocumentXmlZipFileReader
from clinical_trials.clinical_trial_document_mesh_counter import ClinicalTrialDocumentMeshCounter
from mesh.trie.mesh_trie import MeshTrie
from tqdm import tqdm
import numpy as np
import pickle


def build_clinical_trial_mesh_counts(mesh_trie: MeshTrie, clinical_trials_xml_zip_file_path: str):
    ctmc = ClinicalTrialDocumentMeshCounter(mesh_trie)
    with ClinicalTrialDocumentXmlZipFileReader(clinical_trials_xml_zip_file_path) as reader:
        for record in tqdm(reader):
            ctmc.process(record)
    return ctmc


def generate_utility_matrix(ctdmc: ClinicalTrialDocumentMeshCounter, output_file_path: str):
    mesh_index_map = []
    for mesh_index, counter in tqdm(enumerate(ctdmc)):
        if counter is not None:
            mesh_index_map.append(mesh_index)

    num_mesh = len(mesh_index_map)
    num_docs = len(ctdmc.num_processed_docs())

    with open(output_file_path, 'wb') as writer:
        writer.write(np.array([num_mesh, num_docs], dtype='u4').tobytes())
        writer.write(np.array(mesh_index_map, dtype='u4').tobytes())

        for mesh_index in tqdm(mesh_index_map):
            writer.write(np.array(
                [ctdmc.tf_idf(mesh_index, doc_index) for doc_index in range(num_docs)], dtype='f4'
            ).tobytes())



if __name__ == '__main__':
    from mesh.utils import build_mesh_trie

    mesh_trie = build_mesh_trie('../data/d2020.bin')
    ctdmc = build_clinical_trial_mesh_counts(mesh_trie, '../data/Sample.zip')
    with open('../output/Sample.ctdmc', 'wb') as f:
        pickle.dump(ctdmc, f)
