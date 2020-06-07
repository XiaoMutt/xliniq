import typing as tp


class StringIndexer(object):
    def __init__(self):
        self._string_to_index = dict()
        self._strings_array = []

    def add(self, string: str) -> int:
        """
        Add the string and return the index of the string.
        If the string already in the Indexer, directly return the index
        """
        if string in self._string_to_index:
            return self._string_to_index[string]
        else:
            res = len(self._strings_array)
            self._strings_array.append(string)
            self._string_to_index[string] = res
            return res

    def __getitem__(self, item: tp.Union[str, int]):
        if type(item) is str:
            # get index
            return self._string_to_index[item]
        else:
            # get string
            return self._strings_array[item]

    def __len__(self):
        return len(self._strings_array)
