from .builder import IMPUTATION

@IMPUTATION.register_module()
class CustomImputation(object):
    def __init__(self, **kwargs):
        super(CustomImputation, self).__init__()

    def __len__(self):
        """Total number of samples of data."""
        raise NotImplementedError