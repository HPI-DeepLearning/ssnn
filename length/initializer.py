

class Initializer:
    """
    Base structure for all initializers
    """
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, array):
        raise NotImplementedError
