import os
import sys
from pathlib import Path
ROOT = str(Path(__file__).resolve().parents[3])
sys.path.append(ROOT)
import pandas as pd
from ..builder import Market_Dynamics_Model
from ..custom import Market_dynamics_model
from trademaster.utils import get_attr, labeling_util as util
from pathlib import Path
import warnings

@Market_Dynamics_Model.register_module()

class Linear_Market_Dynamics_Model(Market_dynamics_model):
    def __init__(self,**kwargs):
        super(Linear_Market_Dynamics_Model, self).__init__()
        # print('data_path',get_attr(kwargs, "data_path", None))
        self.data_path=get_attr(kwargs, "data_path", None)
        self.method = 'linear'
        self.fitting_parameters = get_attr(kwargs, "fitting_parameters", None)
        self.labeling_parameters = get_attr(kwargs, "labeling_parameters", None)
        self.regime_number = get_attr(kwargs, "regime_number", None)
        self.length_limit = get_attr(kwargs, "length_limit", None)
        self.OE_BTC = get_attr(kwargs, "OE_BTC", None)
        self.PM = get_attr(kwargs, "PM", None)

    def run(self):
        print('labeling start')
        path_names=Path(self.data_path).resolve().parents
        dataset_name=os.path.basename(path_names[0])
        dataset_foler_name=path_names[0]
        task_name = os.path.basename(path_names[1])
        output_path = self.data_path
        if dataset_name=='small_BTC' and task_name=='high_frequency_trading':
            raw_data = pd.read_csv(self.data_path,index_col=0)
            raw_data['tic'] = 'HFT_small_BTC'
            raw_data['adjcp'] = raw_data["close"]
            raw_data['date'] = raw_data.index
            # if not os.path.exists('./temp'):
            #     os.makedirs('./temp')
            process_data_path=os.path.join(dataset_foler_name,dataset_name+'_MDM_processed.csv').replace("\\", "/")
            raw_data.to_csv(process_data_path)
            self.data_path = process_data_path
        if self.OE_BTC == True:
            raw_data = pd.read_csv(self.data_path)
            raw_data['tic'] = 'OE_BTC'
            raw_data['adjcp'] = raw_data["midpoint"]
            try:
                raw_data['date'] = raw_data["system_time"]
            except:
                raw_data['date'] = raw_data["date"]
            # if not os.path.exists('./temp'):
            #     os.makedirs('./temp')
            # raw_data.to_csv('./temp/OE_BTC_processed.csv')
            # self.data_path = './temp/OE_BTC_processed.csv'
            process_data_path=os.path.join(dataset_foler_name,dataset_name+'_MDM_processed.csv').replace("\\", "/")
            raw_data.to_csv(process_data_path)
            self.data_path = process_data_path
        Labeler = util.Labeler(self.data_path, 'linear', self.fitting_parameters)
        print('start fitting')
        Labeler.fit(self.regime_number, self.length_limit)
        print('finish fitting')
        Labeler.label(self.labeling_parameters,os.path.dirname(self.data_path))
        labeled_data = pd.concat([v for v in Labeler.data_dict.values()], axis=0)
        data = pd.read_csv(self.data_path)
        merged_data = data.merge(labeled_data, how='left', on=['date', 'tic', 'adjcp'], suffixes=('', '_DROP')).filter(
            regex='^(?!.*_DROP)')
        low, high = self.labeling_parameters
        self.model_id = str(self.regime_number) + '_' + str(
            self.length_limit) + '_' + str(low) + '_' + str(high)
        if self.PM :
            DJI = merged_data.loc[:, ['date', 'label']]
            test = pd.read_csv(self.PM, index_col=0)
            merged = test.merge(DJI, on='date')
            process_datafile_path = os.path.splitext(output_path)[0] + '_label_by_DJIindex_' + self.model_id + '.csv'
            merged.to_csv(process_datafile_path, index=False)
        else:
            process_datafile_path = os.path.splitext(output_path)[0] + '_labeled_' + self.model_id + '.csv'
            merged_data.to_csv(process_datafile_path
                               , index=False)
        print('labeling done')
        print('plotting start')
        # a list the path to all the modeling visulizations
        market_dynamic_labeling_visualization_paths=Labeler.plot(Labeler.tics, self.labeling_parameters, output_path,self.model_id)
        print('plotting done')
        # if self.OE_BTC == True:
        #     os.remove('./temp/OE_BTC_processed.csv')
        return os.path.abspath(process_datafile_path), market_dynamic_labeling_visualization_paths

