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
        self.test_dynamic_path=osp.join(ROOT, get_attr(kwargs, "test_dynamic_path", None))
        self.transcation_cost=get_attr(kwargs, "transcation_cost", 0.00005)
        self.backward_num_timestamp=get_attr(kwargs, "backward_num_timestamp", 1)
        self.max_holding_number=get_attr(kwargs, "max_holding_number", 0.01)
        self.num_action=get_attr(kwargs, "num_action", 11)
        self.max_punish=get_attr(kwargs, "max_punish", 1e12)
        self.episode_length=get_attr(kwargs, "episode_length", 14400)

        self.tech_indicator_list = get_attr(kwargs, "tech_indicator_list", [])

        test_dynamic=int(get_attr(kwargs, "test_dynamic", "-1"))
        if test_dynamic!=-1:
            self.test_dynamic_paths=[]
            data = pd.read_csv(self.test_dynamic_path)
            data = data.reset_index()
            data = data.loc[data['label'] == test_dynamic, :]
            if data.empty:
                raise ValueError('The there is no market of this style in the test dataset')
            intervals, index_by_tick_list = self.get_styled_intervals_and_gives_new_index(data)
            data.drop(columns=['index'], inplace=True)
            temp_foler=osp.join(ROOT,os.path.dirname(self.test_dynamic_path),'style_slice')
            if not os.path.exists(temp_foler):
                os.makedirs(temp_foler)
            for i, interval in enumerate(intervals):
                data_temp = data.iloc[interval[0]:interval[1], :]
                data_temp.index = index_by_tick_list[i]
                path=osp.join(ROOT,temp_foler,str(test_dynamic) + '_' + str(i) + '.csv')
                data_temp.to_csv(path)
                if max(index_by_tick_list[i]) + 1 <= self.backward_num_timestamp + 2:
                    print('The ' + str(i) + '_th segment length is less than the min length so it won\'t be tested')
                    continue
                self.test_dynamic_paths.append(path)
    def get_styled_intervals_and_gives_new_index(self,data):
        index_by_tick_list = []
        index_by_tick = []
        date = data['date'].to_list()
        last_date = date[0]
        date_counter = 0
        index = data['index'].to_list()
        last_value = index[0] - 1
        last_index = 0
        intervals = []
        for i in range(data.shape[0]):
            if last_value != index[i] - 1:
                date_counter = -1
                intervals.append([last_index, i])
                last_value = index[i]
                last_index = i
                index_by_tick_list.append(index_by_tick)
                index_by_tick = []
            if date[i] != last_date:
                date_counter += 1
            index_by_tick.append(date_counter)
            last_value = index[i]
            last_date = date[i]
        intervals.append([last_index, data.shape[0]])
        index_by_tick_list.append(index_by_tick)
        return intervals, index_by_tick_list