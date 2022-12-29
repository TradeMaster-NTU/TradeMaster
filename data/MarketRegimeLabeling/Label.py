import labeling_util as util
import argparse
import pandas as pd


parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str)
parser.add_argument("--method", type=str)
parser.add_argument("--fitting_parameters",nargs='+', type=str)
parser.add_argument("--labeling_parameters",  nargs="+", type=float)
parser.add_argument('--regime_number',type=int,default=4)
parser.add_argument('--length_limit',type=int,default=0)
args= parser.parse_args()

print('labeling start')
Labeler=util.Labeler(args.data_path,'linear')
Labeler.fit(args.regime_number,args.length_limit)
Labeler.label(args.labeling_parameters)
labeled_data=pd.concat([v for v in Labeler.data_dict.values()],axis=0)
data=pd.read_csv(args.data_path)
merged_data = data.merge(labeled_data,  how='left', on = ['date','tic','adjcp'],suffixes=('', '_DROP')).filter(regex='^(?!.*_DROP)')
merged_data.to_csv(args.data_path[:-4]+'_labeled_'+str(args.regime_number)+'_'+str(args.length_limit)+'.csv', index=False)
print('labeling done')
print('plotting start')
Labeler.plot(Labeler.tics,args.labeling_parameters,args.data_path)
print('plotting done')