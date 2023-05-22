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
        self.dynamic_number = get_attr(kwargs, "dynamic_number", None)
        self.length_limit = get_attr(kwargs, "length_limit", None)
        self.OE_BTC = get_attr(kwargs, "OE_BTC", None)
        self.PM = get_attr(kwargs, "PM", None)
        self.key_indicator = get_attr(kwargs, "key_indicator", None)
        self.timestamp = get_attr(kwargs, "timestamp", None)
        self.tic = get_attr(kwargs, "tic", None)
        self.mode = get_attr(kwargs, "mode", None)
        self.hard_length_limit = get_attr(kwargs, "hard_length_limit", None)
        self.slope_diff_threshold=get_attr(kwargs, "slope_diff_threshold", None)

    def file_extension_selector(self,read):
        if self.data_path.endswith('.csv'):
            if read:
                return pd.read_csv
            else:
                return pd.DataFrame.to_feather
        elif self.data_path.endswith('.feather'):
            if read:
                return pd.read_feather
            else:
                return pd.DataFrame.to_feather
        else:
            raise ValueError('invalid file extension')

    def run(self):
        print('labeling start')
        path_names=Path(self.data_path).resolve().parents
        dataset_name=os.path.basename(path_names[0])
        dataset_foler_name=path_names[0]
        task_name = os.path.basename(path_names[1])
        output_path = self.data_path
        if dataset_name=='small_BTC' and task_name=='high_frequency_trading':
            raw_data = pd.read_csv(self.data_path,index_col=0)
            raw_data[self.tic] = 'HFT_small_BTC'
            raw_data[self.key_indicator] = raw_data["close"]
            raw_data[self.timestamp] = raw_data.index
            # if not os.path.exists('./temp'):
            #     os.makedirs('./temp')
            process_data_path=os.path.join(dataset_foler_name,dataset_name+'_MDM_processed.csv').replace("\\", "/")
            raw_data.to_csv(process_data_path)
            self.data_path = process_data_path
        if self.OE_BTC == True:
            raw_data = pd.read_csv(self.data_path)
            raw_data[self.tic] = 'OE_BTC'
            raw_data[self.key_indicator] = raw_data["midpoint"]
            try:
                raw_data[self.timestamp] = raw_data["system_time"]
            except:
                raw_data[self.timestamp] = raw_data["date"]
            # if not os.path.exists('./temp'):
            #     os.makedirs('./temp')
            # raw_data.to_csv('./temp/OE_BTC_processed.csv')
            # self.data_path = './temp/OE_BTC_processed.csv'
            process_data_path=os.path.join(dataset_foler_name,dataset_name+'_MDM_processed.csv').replace("\\", "/")
            raw_data.to_csv(process_data_path)
            self.data_path = process_data_path
        Labeler = util.Labeler(self.data_path, 'linear', self.fitting_parameters,key_indicator=self.key_indicator,
                               timestamp=self.timestamp, tic=self.tic, mode=self.mode,hard_length_limit=self.hard_length_limit,
                               slope_diff_threshold=self.slope_diff_threshold)
        print('start fitting')
        Labeler.fit(self.dynamic_number, self.length_limit, self.hard_length_limit)
        print('finish fitting')
        Labeler.label(self.labeling_parameters,os.path.dirname(self.data_path))
        labeled_data = pd.concat([v for v in Labeler.data_dict.values()], axis=0)
        # file_writer=self.file_extension_selector(read=False)
        # print('file_writer',file_writer)
        flie_reader=self.file_extension_selector(read=True)
        extension=self.data_path.split('.')[-1]
        data=flie_reader(self.data_path)
        if self.tic in data.columns:
            merge_keys = [self.timestamp, self.tic, self.key_indicator]
        else:
            merge_keys = [self.timestamp, self.key_indicator]
        merged_data = data.merge(labeled_data, how='left', on=merge_keys, suffixes=('', '_DROP')).filter(
            regex='^(?!.*_DROP)')
        if self.mode=='slope':
            low, high = self.labeling_parameters
            self.model_id = str(self.dynamic_number) + '_' + str(
                self.length_limit) + '_' + str(low) + '_' + str(high)
        elif self.mode=='quantile':
            self.model_id = f"{self.dynamic_number}_{self.length_limit}_quantile"
        if self.PM :
            DJI = merged_data.loc[:, [self.timestamp, 'label']]
            test = pd.read_csv(self.PM, index_col=0)
            merged = test.merge(DJI, on=self.timestamp)
            process_datafile_path = os.path.splitext(output_path)[0] + '_label_by_DJIindex_' + self.model_id + '.'+ extension
            merged_data.to_csv(process_datafile_path, index=False)
        else:
            process_datafile_path = os.path.splitext(output_path)[0] + '_labeled_' + self.model_id +'.'+ extension
            if extension == 'csv':
                merged_data.to_csv(process_datafile_path, index=False)
            elif extension == 'feather':
                merged_data.to_feather(process_datafile_path)
        print('labeling done')
        print('plotting start')
        # a list the path to all the modeling visulizations
        market_dynamic_labeling_visualization_paths=Labeler.plot(Labeler.tics, self.labeling_parameters, output_path,self.model_id)
        print('plotting done')
        # if self.OE_BTC == True:
        #     os.remove('./temp/OE_BTC_processed.csv')
        return os.path.abspath(process_datafile_path), market_dynamic_labeling_visualization_paths

