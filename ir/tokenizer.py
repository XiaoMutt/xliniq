import re
from enum import Enum
import typing as tp


class Tokenizer(object):
    class SPLIT_PATTERN(Enum):
        # tokenize text
        TEXT_SPLIT_PATTERN = re.compile(r'[\W\n]')
        # for keyword, comma is to put important word in the beginning
        KEYWORD_SPLIT_PATTERN = re.compile(r'[^[a-zA-Z0-9_,]')
        # tokenize numbers
        NUMBER_SPLIT_PATTERN = re.compile(r'[^\d]')

    def __init__(self, split_pattern: SPLIT_PATTERN):
        self._split_pattern = split_pattern.value

    def __call__(self, string: tp.Union[str, None]) -> tuple:
        """
        split a string to tokens according to the split pattern.
        :param string: the string to split
        :return: tuple
        """
        if type(string) == str:
            return tuple(filter(lambda x: x != '', self._split_pattern.split(string)))  # get rid of empty string
        else:
            return tuple()
