import os
import random
import sys
from pathlib import Path
ROOT = str(Path(__file__).resolve().parents[3])
sys.path.append(ROOT)
import argparse
import pandas as pd
from ..builder import Market_Dynamics_Model
from ..custom import Market_dynamics_model
from trademaster.utils import get_attr, labeling_util as util


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
        output_path = self.data_path
        if self.OE_BTC == True:
            raw_data = pd.read_csv(self.data_path)
            raw_data['tic'] = 'OE_BTC'
            raw_data['adjcp'] = raw_data["midpoint"]
            raw_data['date'] = raw_data["system_time"]
            if not os.path.exists('./temp'):
                os.makedirs('./temp')
            raw_data.to_csv('./temp/OE_BTC_processed.csv')
            self.data_path = './temp/OE_BTC_processed.csv'
        Labeler = util.Labeler(self.data_path, 'linear', self.fitting_parameters)
        Labeler.fit(self.regime_number, self.length_limit)
        Labeler.label(self.labeling_parameters)
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
        print(output_path)
        market_dynamic_labeling_visualization_paths=Labeler.plot(Labeler.tics, self.labeling_parameters, output_path,self.model_id)
        print('plotting done')
        if self.OE_BTC == True:
            os.remove('./temp/OE_BTC_processed.csv')
        return process_datafile_path, market_dynamic_labeling_visualization_paths



def main(args):
    print('labeling start')
    output_path = args.data_path
    if args.OE_BTC == True:
        raw_data = pd.read_csv(args.data_path)
        raw_data['tic'] = 'OE_BTC'
        raw_data['adjcp'] = raw_data["midpoint"]
        raw_data['date'] = raw_data["system_time"]
        if not os.path.exists('./temp'):
            os.makedirs('./temp')
        raw_data.to_csv('./temp/OE_BTC_processed.csv')
        args.data_path = './temp/OE_BTC_processed.csv'
    Labeler = util.Labeler(args.data_path, 'linear',args.fitting_parameters)
    Labeler.fit(args.regime_number, args.length_limit)
    Labeler.label(args.labeling_parameters)
    labeled_data = pd.concat([v for v in Labeler.data_dict.values()], axis=0)
    data = pd.read_csv(args.data_path)
    merged_data = data.merge(labeled_data, how='left', on=['date', 'tic', 'adjcp'], suffixes=('', '_DROP')).filter(
        regex='^(?!.*_DROP)')
    low, high = args.labeling_parameters
    if args.PM :
        DJI = merged_data.loc[:, ['date', 'label']]
        test = pd.read_csv(args.PM, index_col=0)
        merged = test.merge(DJI, on='date')
        process_datafile_path=output_path[:-4] + '_label_by_DJIindex_' + str(args.regime_number) + '_' + str(
            args.length_limit) + '_' + str(low) + '_' + str(high) + '.csv'
        merged.to_csv(process_datafile_path, index=False)
    else:
        process_datafile_path=output_path[:-4] + '_labeled_' + str(args.regime_number) + '_' + str(args.length_limit) + '_' + str(
                low) + '_' + str(high) + '.csv'
        merged_data.to_csv(process_datafile_path
            , index=False)
    print('labeling done')
    print('plotting start')
    # a list the path to all the modeling visulizations
    market_dynamic_labeling_visulization_paths=[]
    market_dynamic_labeling_visulization_paths.append(Labeler.plot(Labeler.tics, args.labeling_parameters, output_path))
    print('plotting done')
    if args.OE_BTC == True:
        os.remove('./temp/OE_BTC_processed.csv')
    return process_datafile_path,market_dynamic_labeling_visulization_paths



if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--method", type=str)
    parser.add_argument("--fitting_parameters",nargs='+', type=str)
    parser.add_argument("--labeling_parameters",  nargs="+", type=float)
    parser.add_argument('--regime_number',type=int,default=4)
    parser.add_argument('--length_limit',type=int,default=0)
    parser.add_argument('--OE_BTC',type=bool,default=False)
    parser.add_argument('--PM',type=str,default='')
    args= parser.parse_args()
    main(args)