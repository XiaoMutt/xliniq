from mesh.trie.mesh_trie import MeshTrie
from mesh.file_reader.descriptor_file_reader import DescriptorAscIIFileReader
from tqdm import tqdm


def build_mesh_trie(file_path: str) -> MeshTrie:
    res = MeshTrie()
    with DescriptorAscIIFileReader(file_path) as reader:
        for record in tqdm(reader):
            res.add(record)
    return res
