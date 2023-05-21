import os
import sys
from pathlib import Path
ROOT = str(Path(__file__).resolve().parents[4])
sys.path.append(ROOT)
import argparse
import pandas as pd
from trademaster.utils import  labeling_util as util
from pathlib import Path
import warnings


def main(args):
    warnings.warn("running this script as main is deprecated, you should run the run.py in "
                  "tool/market_dynamics_labeling/ to use the latest version ")
    print('labeling start')
    path_names = Path(args.data_path).resolve().parents
    dataset_name = os.path.basename(path_names[0])
    dataset_foler_name = path_names[0]
    task_name = os.path.basename(path_names[1])
    output_path = args.data_path
    if dataset_name == 'small_BTC' and task_name == 'high_frequency_trading':
        raw_data = pd.read_csv(args.data_path, index_col=0)
        raw_data[args.tic] = 'HFT_small_BTC'
        raw_data[args.key_indicator] = raw_data["close"]
        raw_data[args.timestamp] = raw_data.index
        process_data_path = os.path.join(dataset_foler_name, dataset_name + '_MDM_processed.csv').replace("\\", "/")
        raw_data.to_csv(process_data_path)
        args.data_path = process_data_path
    if args.OE_BTC == True:
        raw_data = pd.read_csv(args.data_path)
        raw_data[args.tic] = 'OE_BTC'
        raw_data[args.key_indicator] = raw_data["midpoint"]
        raw_data[args.timestamp] = raw_data["system_time"]
        process_data_path = os.path.join(dataset_foler_name, dataset_name + '_MDM_processed.csv').replace("\\", "/")
        raw_data.to_csv(process_data_path)
        args.data_path = process_data_path
    Labeler = util.Labeler(args.data_path, 'linear',args.fitting_parameters,key_indicator=args.key_indicator,timestamp=args.timestamp,tic=args.tic)
    Labeler.fit(args.regime_number, args.length_limit)
    Labeler.label(args.labeling_parameters,os.path.dirname(args.data_path))
    labeled_data = pd.concat([v for v in Labeler.data_dict.values()], axis=0)
    data = pd.read_csv(args.data_path)
    merged_data = data.merge(labeled_data, how='left', on=[args.timestamp, args.tic, args.key_indicator], suffixes=('', '_DROP')).filter(
        regex='^(?!.*_DROP)')
    low, high = args.labeling_parameters
    model_id = str(args.regime_number) + '_' + str(
        args.length_limit) + '_' + str(low) + '_' + str(high)
    if args.PM:
        DJI = merged_data.loc[:, [args.timestamp, 'label']]
        test = pd.read_csv(args.PM, index_col=0)
        merged = test.merge(DJI, on=args.timestamp)
        process_datafile_path = os.path.splitext(output_path)[0] + '_label_by_DJIindex_' + model_id + '.csv'
        merged.to_csv(process_datafile_path, index=False)
    else:
        process_datafile_path = os.path.splitext(output_path)[0] + '_labeled_' + model_id + '.csv'
        merged_data.to_csv(process_datafile_path
                           , index=False)
    print('labeling done')
    print('plotting start')
    # a list the path to all the modeling visulizations
    market_dynamic_labeling_visualization_paths = Labeler.plot(Labeler.tics, args.labeling_parameters, output_path,
                                                               model_id)
    print('plotting done')
    print(f'the processed datafile is at {process_datafile_path}')
    plot_dir = os.path.dirname(os.path.realpath(market_dynamic_labeling_visualization_paths[0]))
    print(f'the visualizations are at {plot_dir}')
    return os.path.abspath(process_datafile_path), market_dynamic_labeling_visualization_paths

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
    parser.add_argument("--key_indicator", type=str, default='adjcp')
    parser.add_argument("--timestamp", type=str, default='timestamp')
    parser.add_argument("--tic", type=str, default='tic')
    args= parser.parse_args()
    main(args)