from base.immutable import Immutable
import typing as tp


class MeshSupplementaryRecord(Immutable):
    heading: str
    entries: tuple
    mapped_to: tuple

    def __init__(self, heading: str, entries: tp.Iterable, numbers: tp.Iterable):
        super(MeshSupplementaryRecord, self).__init__(heading=heading,
                                                      entries=tuple(entries),
                                                      mapped_to=tuple(numbers))
