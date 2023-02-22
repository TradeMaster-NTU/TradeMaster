from .builder import PREPROCESSOR

@PREPROCESSOR.register_module()
class CustomPreprocessor(object):
    def __init__(self, **kwargs):
        super(CustomPreprocessor, self).__init__()

    def __len__(self):
        """Total number of samples of data."""
        raise NotImplementedError