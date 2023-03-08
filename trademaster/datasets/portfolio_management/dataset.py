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
class PortfolioManagementDataset(CustomDataset):
    def __init__(self, **kwargs):
        super(PortfolioManagementDataset, self).__init__()

        self.kwargs = kwargs

        self.data_path = osp.join(ROOT, get_attr(kwargs, "data_path", None))

        self.train_path = osp.join(ROOT, get_attr(kwargs, "train_path", None))
        self.valid_path = osp.join(ROOT, get_attr(kwargs, "valid_path", None))
        self.test_path = osp.join(ROOT, get_attr(kwargs, "test_path", None))
        self.test_dynamic_path=osp.join(ROOT, get_attr(kwargs, "test_dynamic_path", None))
        test_dynamic=int(get_attr(kwargs, "test_dynamic", "-1"))
        if test_dynamic!=-1:
            length_day= get_attr(kwargs, "length_day", 0)
            self.test_dynamic_paths=[]
            data = pd.read_csv(self.test_dynamic_path)
            data = data.reset_index()
            ## vote for data['label'] for multiple labels by date
            voter = data.loc[:, ["date", "label"]].groupby(["date", "label"], as_index=False).size()
            voter = voter.groupby(["date"]).apply(lambda x: x.nlargest(1, ['size'])).reset_index(drop=True)
            data = data.merge(voter.loc[:, ["date", "label"]], left_on='date', right_on='date')
            data['label'] = data['label_y']
            data.drop(['label_x', 'label_y'], axis=1, inplace=True)

            temp_foler=osp.join(ROOT,os.path.dirname(self.test_dynamic_path),'style_slice')
            if not os.path.exists(temp_foler):
                os.makedirs(temp_foler)
            data.to_csv(osp.join(ROOT, temp_foler, 'process_data.csv'))
            data = data.loc[data['label'] == test_dynamic, :]

            intervals, index_by_tick_list = self.get_styled_intervals_and_gives_new_index(data)
            data.drop(columns=['index'], inplace=True)




            for i, interval in enumerate(intervals):
                data_temp = data.iloc[interval[0]:interval[1], :]
                data_temp.index = index_by_tick_list[i]
                path=osp.join(ROOT,temp_foler,str(test_dynamic) + '_' + str(i) + '.csv')
                data_temp.to_csv(path)
                if max(index_by_tick_list[i]) + 1 <= length_day:
                    print('The ' + str(i) + '_th segment length is less than the min length so it won\'t be tested')
                    continue
                self.test_dynamic_paths.append(path)

        self.tech_indicator_list = get_attr(kwargs, "tech_indicator_list", [])
        self.initial_amount = get_attr(kwargs, "initial_amount", 100000)
        self.length_day = get_attr(kwargs, "length_day", 10)
        self.transaction_cost_pct = get_attr(kwargs, "transaction_cost_pct", 0.001)

    def get_styled_intervals_and_gives_new_index(self, data):
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