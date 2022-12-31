import labeling_util as util
import argparse
import pandas as pd
import os

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str)
parser.add_argument("--method", type=str)
parser.add_argument("--fitting_parameters",nargs='+', type=str)
parser.add_argument("--labeling_parameters",  nargs="+", type=float)
parser.add_argument('--regime_number',type=int,default=4)
parser.add_argument('--length_limit',type=int,default=0)
parser.add_argument('--OE_BTC',type=bool,default=False)
args= parser.parse_args()

print('labeling start')
output_path=args.data_path
if args.OE_BTC==True:
    raw_data=pd.read_csv(args.data_path)
    raw_data['tic']='OE_BTC'
    raw_data['adjcp']=raw_data["midpoint"]
    raw_data['date'] = raw_data["system_time"]
    if not os.path.exists('./temp'):
        os.makedirs('./temp')
    raw_data.to_csv('./temp/OE_BTC_processed.csv')
    args.data_path='./temp/OE_BTC_processed.csv'
Labeler=util.Labeler(args.data_path,'linear')
Labeler.fit(args.regime_number,args.length_limit)
Labeler.label(args.labeling_parameters)
labeled_data=pd.concat([v for v in Labeler.data_dict.values()],axis=0)
data=pd.read_csv(args.data_path)
merged_data = data.merge(labeled_data,  how='left', on = ['date','tic','adjcp'],suffixes=('', '_DROP')).filter(regex='^(?!.*_DROP)')
merged_data.to_csv(output_path[:-4]+'_labeled_'+str(args.regime_number)+'_'+str(args.length_limit)+'.csv', index=False)
print('labeling done')
print('plotting start')
Labeler.plot(Labeler.tics,args.labeling_parameters,output_path)
print('plotting done')
if args.OE_BTC==True:
    os.remove('./temp/OE_BTC_processed.csv')