

class AbstractInitializer:
    """
    Abstractly initializers nothing
    """
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, array):
        raise NotImplementedError
