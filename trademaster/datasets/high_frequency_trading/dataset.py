from pathlib import Path
import sys
ROOT = str(Path(__file__).resolve().parents[3])
sys.path.append(ROOT)

import os.path as osp
from ..custom import CustomDataset
from ..builder import DATASETS
from trademaster.utils import get_attr
import pandas as pd
import os

@DATASETS.register_module()
class HighFrequencyTradingDataset(CustomDataset):
    def __init__(self, **kwargs):
        super(HighFrequencyTradingDataset, self).__init__()

        self.kwargs = kwargs

        self.data_path = osp.join(ROOT, get_attr(kwargs, "data_path", None))

        self.train_path = osp.join(ROOT, get_attr(kwargs, "train_path", None))
        self.valid_path = osp.join(ROOT, get_attr(kwargs, "valid_path", None))
        self.test_path = osp.join(ROOT, get_attr(kwargs, "test_path", None))
        self.test_style_path=osp.join(ROOT, get_attr(kwargs, "test_style_path", None))
        self.transcation_cost=get_attr(kwargs, "transcation_cost", 0.00005)
        self.backward_num_timestamp=get_attr(kwargs, "backward_num_timestamp", 1)
        self.max_holding_number=get_attr(kwargs, "max_holding_number", 0.01)
        self.num_action=get_attr(kwargs, "num_action", 11)
        self.max_punish=get_attr(kwargs, "max_punish", 1e12)
        self.episode_length=get_attr(kwargs, "episode_length", 14400)

        self.tech_indicator_list = get_attr(kwargs, "tech_indicator_list", [])
