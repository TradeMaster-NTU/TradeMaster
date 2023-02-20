from .builder import DATASETS
from torch.utils.data import Dataset

@DATASETS.register_module()
class CustomDataset(Dataset):
    def __init__(self, **kwargs):
        super(CustomDataset, self).__init__()

    def __len__(self):
        """Total number of samples of data."""
        raise NotImplementedError