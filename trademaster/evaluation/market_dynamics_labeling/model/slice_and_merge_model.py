import os
import sys
from pathlib import Path
ROOT = str(Path(__file__).resolve().parents[3])
sys.path.append(ROOT)
import pandas as pd
from ..builder import Market_Dynamics_Model
from ..custom import Market_dynamics_model
from trademaster.utils import get_attr, labeling_util as util,market_dynamics_modeling_analysis
from pathlib import Path

@Market_Dynamics_Model.register_module()

class Linear_Market_Dynamics_Model(Market_dynamics_model):
    def __init__(self,**kwargs):
        super(Linear_Market_Dynamics_Model, self).__init__()
        self.data_path=get_attr(kwargs, "data_path", None)
        self.method = 'slice_and_merge'
        self.filter_strength = get_attr(kwargs, "filter_strength", None)
        self.slope_interval = get_attr(kwargs, "slope_interval", None)
        self.dynamic_number = get_attr(kwargs, "dynamic_number", None)
        self.max_length_expectation = get_attr(kwargs, "max_length_expectation", None)
        self.OE_BTC = get_attr(kwargs, "OE_BTC", None)
        self.PM = get_attr(kwargs, "PM", None)
        self.key_indicator = get_attr(kwargs, "key_indicator", None)
        self.timestamp = get_attr(kwargs, "timestamp", None)
        self.tic = get_attr(kwargs, "tic", None)
        self.labeling_method = get_attr(kwargs, "labeling_method", None)
        self.min_length_limit = get_attr(kwargs, "min_length_limit", None)
        self.merging_metric=get_attr(kwargs, "merging_metric", None)
        self.merging_threshold=get_attr(kwargs, "merging_threshold", None)
        self.merging_dynamic_constraint=get_attr(kwargs, "merging_dynamic_constraint", None)
        self.exp_name=get_attr(kwargs, "exp_name", None)

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

    def wirte_data_as_segments(self,data,process_datafile_path):
        # get file name and extension from process_datafile_path
        file_name, file_extension = os.path.splitext(process_datafile_path)


    def run(self):
        path_names=Path(self.data_path).resolve().parents
        dataset_name=os.path.basename(path_names[0])
        dataset_foler_name=path_names[0]
        task_name = os.path.basename(path_names[1])
        # create a folder by tic name to store the outputs
        # get the folder name of self.data_path
        folder_name=os.path.dirname(self.data_path)
        # file name without extension
        file_name=os.path.splitext(os.path.basename(self.data_path))[0]
        output_path=os.path.join(folder_name,self.exp_name,self.tic).replace("\\", "/")
        # make output_path for tic
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        if dataset_name=='small_BTC' and task_name=='high_frequency_trading':
            raw_data = pd.read_csv(self.data_path,index_col=0)
            raw_data[self.tic] = 'HFT_small_BTC'
            # raw_data[self.key_indicator] = raw_data["close"]
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
        worker = util.Worker(data_path=self.data_path, method='slice_and_merge', filter_strength=self.filter_strength, key_indicator=self.key_indicator,
                             timestamp=self.timestamp, tic=self.tic, labeling_method=self.labeling_method, min_length_limit=self.min_length_limit,
                             merging_threshold=self.merging_threshold, merging_metric=self.merging_metric,merging_dynamic_constraint=self.merging_dynamic_constraint)
        print('----------------------------------')
        print('start fitting')
        worker.fit(self.dynamic_number, self.max_length_expectation, self.min_length_limit)
        print('fitting done')
        print('----------------------------------')
        print('start labeling')
        worker.label(self.slope_interval, os.path.dirname(self.data_path))
        labeled_data = pd.concat([v for v in worker.data_dict.values()], axis=0)
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
        if self.labeling_method=='slope':
            low, high = self.slope_interval
            self.model_id =  f"slice_and_merge_model_{self.dynamic_number}dynamics_minlength{self.min_length_limit}_{self.labeling_method}_labeling_slope{low}_{high}"
        else:
            self.model_id = f"slice_and_merge_model_{self.dynamic_number}dynamics_minlength{self.min_length_limit}_{self.labeling_method}_labeling"
        if self.PM :
            DJI = merged_data.loc[:, [self.timestamp, 'label']]
            test = pd.read_csv(self.PM, index_col=0)
            merged = test.merge(DJI, on=self.timestamp)
            process_datafile_path = os.path.join(output_path,
                                                 file_name + '_label_by_DJIindex_' + self.model_id + '.' + extension).replace(
                "\\", "/")
            merged_data.to_csv(process_datafile_path, index=False)
        else:
            process_datafile_path = os.path.join(output_path,file_name+ '_labeled_' + self.model_id +'.'+ extension).replace("\\", "/")
            if extension == 'csv':
                merged_data.to_csv(process_datafile_path, index=False)
            elif extension == 'feather':
                merged_data.to_feather(process_datafile_path)
        print('labeling done')
        print('----------------------------------')
        print('start plotting')
        # a list the path to all the modeling visulizations
        market_dynamic_modeling_visualization_paths=worker.plot(worker.tics, self.slope_interval, output_path, self.model_id)
        print('plotting done')
        print('----------------------------------')
        # if self.OE_BTC == True:
        #     os.remove('./temp/OE_BTC_processed.csv')

        #MDM analysis
        print('start market dynamics modeling analysis')
        MDM_analysis=market_dynamics_modeling_analysis.MarketDynamicsModelingAnalysis(process_datafile_path,self.key_indicator)
        market_dynamic_modeling_analysis_paths=MDM_analysis.run_analysis(process_datafile_path)
        print('market dynamics modeling analysis done')

        return os.path.abspath(process_datafile_path), market_dynamic_modeling_visualization_paths,market_dynamic_modeling_analysis_paths

