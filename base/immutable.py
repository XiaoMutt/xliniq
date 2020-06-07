class Immutable(object):
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            super(Immutable, self).__setattr__(key, value)

    def __setattr__(self, key: str, value):
        raise Exception(f'Object {self.__class__} is immutable: cannot set {key} to {value}.')
