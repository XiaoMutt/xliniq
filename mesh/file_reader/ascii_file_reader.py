from base.file_reader import FileReader
from abc import ABC


class AscIIFileReader(FileReader, ABC):
    def __init__(self, file_path: str):
        super(AscIIFileReader, self).__init__(file_path)
        self._line = ''

    def __enter__(self):
        if self._file_handler is None:
            self._file_handler = open(self._file_path, 'r')
        return self
