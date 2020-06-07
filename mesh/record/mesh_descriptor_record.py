from base.immutable import Immutable
import typing as tp


class MeshDescriptorRecord(Immutable):
    heading: str
    entries: tuple
    numbers: tuple

    def __init__(self, heading: str, entries: tp.Iterable, numbers: tp.Iterable):
        super(MeshDescriptorRecord, self).__init__(heading=heading,
                                                   entries=tuple(entries),
                                                   numbers=tuple(numbers))
