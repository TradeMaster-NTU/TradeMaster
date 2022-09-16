import pickle
import os
import pandas as pd
import torch
import pdb
import argparse

parser = argparse.ArgumentParser(description="CSDI")
parser.add_argument("--dataset", type=str, default="dj30")
parser.add_argument("--tic", type=str, default="AAPL")
parser.add_argument("--testmissingratio", type=float, default=0.1)
args = parser.parse_args()

path = "./data/" + args.dataset + "/" + args.tic + "_missing" + str(args.testmissingratio) + "_seed1.pk"
path2 = "./save/" + args.dataset + "/" + args.tic + "_" + str(args.testmissingratio) + "/generated_outputs_nsample100.pk"



def get_quantile(samples,q,dim=1):
    return torch.quantile(samples,q,dim=dim).cpu().numpy()


with open(path, "rb") as f:
                observed_values, observed_masks, gt_masks, mean, std = pickle.load(f)

with open(path2, 'rb') as f:
    samples,all_target,all_evalpoint,all_observed,all_observed_time,scaler,mean_scaler = pickle.load(f)


all_target_np = all_target.cpu().numpy()
all_evalpoint_np = all_evalpoint.cpu().numpy()
all_observed_np = all_observed.cpu().numpy()
all_given_np = all_observed_np - all_evalpoint_np
K = samples.shape[-1] #feature
L = samples.shape[-2] #time length

for k in range(K):
    samples[:,:,:,k] = samples[:,:,:,k]*std[k]+mean[k]
    all_target_np[:,:,k] = all_target_np[:,:,k]*std[k]+mean[k]

qlist =[0.05,0.25,0.5,0.75,0.95]
quantiles_imp= []
for q in qlist:
    quantiles_imp.append(get_quantile(samples, q, dim=1)*(1-all_given_np) + all_target_np * all_given_np)

dataset_path = "./source/" + args.dataset + ".csv"
data = pd.read_csv(dataset_path)
date_list = data.date.unique().tolist()
start_date = date_list[0]
end_date = date_list[-1]
date_list_2 = pd.bdate_range(start=start_date, end=end_date)
df = data.loc[data['tic']==args.tic]
df = df[['date', 'open', 'high', 'low', 'close', 'adjcp']]
df = df.sort_values(by=["date"])
date = [ x.strftime('%F') for x in date_list_2]
df_new = pd.DataFrame(columns=['date', 'open', 'high', 'low', 'close', 'adjcp'])
df_impute = pd.DataFrame(index=range(1), columns=['date', 'open', 'high', 'low', 'close', 'adjcp'])

impute_list = []
count = 0
for d in date:
    if d in date_list:
        df_new = pd.concat([df_new,df[df["date"]==d]])
    else:
        df_impute['date'] = d
        if count <= samples.shape[0]*10:
            impute_list.append(count)
            df_impute['open'] = quantiles_imp[2][count//10,count%10,0]
            df_impute['high'] = quantiles_imp[2][count//10,count%10,1]
            df_impute['low'] = quantiles_imp[2][count//10,count%10,2]
            df_impute['close'] = quantiles_imp[2][count//10,count%10,3]
            df_impute['adjcp'] = quantiles_imp[2][count//10,count%10,4]
            df_new = pd.concat([df_new,df_impute])
    count+=1
df_new = df_new.sort_values(by=["date"]).reset_index(drop=True)
print(impute_list)
path3 = './imputed/' + args.dataset
os.makedirs(path3, exist_ok=True)
df_new.to_csv(path3 + "/" + args.tic + '.csv')




