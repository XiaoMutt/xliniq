from abc import abstractmethod


class FileReader(object):
    def __init__(self, file_path: str):
        if file_path is None:
            raise Exception(f"mesh_file_name not provided.")
        self._file_path = file_path
        self._file_handler = None

    @abstractmethod
    def __enter__(self):
        if self._file_handler is None:
            raise Exception("File Handler is unimplemented yet.")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._file_handler.close()

    def __del__(self):
        if self._file_handler is not None:
            self._file_handler.close()

    def __iter__(self):
        if self._file_handler is None:
            self.__enter__()
        else:
            self._file_handler.seek(0)
        return self

    @abstractmethod
    def __next__(self):
        raise Exception("Unimplemented yet.")
