import labeling_util as util
import argparse
import pandas as pd


parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str)
parser.add_argument("--method", type=str)
parser.add_argument("--fitting_parameters",nargs='+', type=str)
parser.add_argument("--labeling_parameters",  nargs="+", type=float)
args= parser.parse_args()

print('labeling start')
Labeler=util.Labeler(args.data_path,'linear')
Labeler.fit()
Labeler.label(args.labeling_parameters)
labeled_data=pd.concat([v for v in Labeler.data_dict.values()],axis=0)
data=pd.read_csv(args.data_path)
merged_data = data.merge(labeled_data,  how='left', on = ['date','tic','adjcp'])
merged_data.to_csv(args.data_path[:-4]+'_labeled.csv')
print('labeling done')
Labeler.plot(Labeler.tics,args.labeling_parameters)
print('plotting done')