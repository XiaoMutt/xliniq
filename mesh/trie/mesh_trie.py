import typing as tp
from mesh.record.mesh_descriptor_record import MeshDescriptorRecord
from ir.string_indexer import StringIndexer
from ir.tokenizer import Tokenizer
from collections import Counter


class MeshTrie(object):

    def __init__(self):
        self._root = dict()  # store mesh tries: token->token->...->'#'->{index1, index2, ...}
        self._mesh_indexer = StringIndexer()

        # tokenizer used to tokenize keywords
        self._keyword_tokenizer = Tokenizer(Tokenizer.SPLIT_PATTERN.KEYWORD_SPLIT_PATTERN)

        # tokenizer used to tokenize text
        self._text_tokenizer = Tokenizer(Tokenizer.SPLIT_PATTERN.TEXT_SPLIT_PATTERN)

    @property
    def total_meshes(self):
        return len(self._mesh_indexer)

    def _add_tokens(self, tokens: tp.Iterable, index: int):
        current = self._root
        for token in tokens:
            if token not in current:
                current[token] = dict()
            current = current[token]

        if current != self._root:
            if '#' not in current:
                current['#'] = {index}
            elif index not in current['#']:
                current['#'] = set(current['#'])  # make a copy
                current['#'].add(index)

    def add(self, descriptor_record: MeshDescriptorRecord) -> int:
        """
        Add MeshDescriptorRecord to the Trie and return the index of the heading.
        """
        # convert heading to index
        index = self._mesh_indexer.add(descriptor_record.heading)
        # add entries to the trie
        for entry in descriptor_record.entries:
            tokens = self._keyword_tokenizer(entry)
            self._add_tokens(tokens, index)
        # add heading to the trie
        self._add_tokens(self._keyword_tokenizer(descriptor_record.heading), index)
        return index

    def get_index(self, tokens: tp.Iterable) -> int:
        # raise exception if token not in the trie
        current = self._root
        for token in tokens:
            current = self._root[token]
        return current['#']

    def get_heading(self, tokens: tp.Iterable) -> str:
        # raise exception if token not in the trie
        idx = self.get_index(tokens)
        return self._mesh_indexer[idx]

    def count_mesh_indices(self, text: str) -> Counter:
        def lazy_mesh_finder(start) -> int:
            """
            Find the shortest meshes in tokens starting from the start position. Add it to res and return the position
            of the token that does not match the MeshTrie .
            If no meshes are found, return the start+1.
            """
            pos = start
            current = self._root
            while pos < len(tokens):
                token = tokens[pos]
                pos += 1
                if token in current:
                    current = current[token]

                    if '#' in current:
                        indices = current['#']
                        c = 1 / len(indices)  # if there are several indices, split 1 evenly over them
                        for mesh_idx in indices:
                            res[mesh_idx] += c
                        return pos
                else:
                    break

            return start + 1

        res = Counter()  # mesh_index-> count
        tokens = self._text_tokenizer(text)

        idx = 0
        while idx < len(tokens):
            idx = lazy_mesh_finder(idx)
        return res

    def count_meshes(self, text: str) -> Counter:
        res = Counter()
        for mesh_idx, count in self.count_mesh_indices(text).items():
            res[self._mesh_indexer[mesh_idx]] = count
        return res
