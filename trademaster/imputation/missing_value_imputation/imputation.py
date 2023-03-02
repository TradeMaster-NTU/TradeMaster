from pathlib import Path
import sys

ROOT = str(Path(__file__).resolve().parents[3])
sys.path.append(ROOT)

import os
import os.path as osp
from ..custom import CustomImputation
from ..builder import IMPUTATION
from trademaster.utils import get_attr

import argparse
import torch
import json
import yaml
import matplotlib.pyplot as plt
import mplfinance as mpf
import pandas as pd
import numpy as np
import pickle

from trademaster.imputation.missing_value_imputation.model.main_model import CSDI_own
from trademaster.imputation.missing_value_imputation.dataset import get_dataloader
from trademaster.imputation.missing_value_imputation.utils import train, evaluate


@IMPUTATION.register_module()
class CSDI_Imputation(CustomImputation):
    def __init__(self, **kwargs):
        super(CSDI_Imputation, self).__init__()
        self.kwargs = kwargs
        self.seed = get_attr(kwargs,"seed", 1)
        self.batch_size = get_attr(kwargs,"batch_size", None)
        self.missing_ratio = get_attr(kwargs,"missing_ratio", None)
        self.dataset_name = get_attr(kwargs,"dataset_name", None)
        self.tic_name = get_attr(kwargs,"tic_name", None)
        self.device = "cuda"
        self.nsample = get_attr(kwargs,"nsample", None)
        self.modelfolder = "./work_dir/missing_value_imputation/save/" + self.dataset_name + "/"+ self.tic_name + "_" + str(self.missing_ratio) + "/"
        self.datafolder = "./work_dir/missing_value_imputation/data/" + self.dataset_name + "/"
        os.makedirs(self.datafolder, exist_ok=True)


    def data_loader(self):
        train_loader, valid_loader, test_loader = get_dataloader(
        seed = self.seed,
        batch_size = self.batch_size,
        missing_ratio = self.missing_ratio,
        dataset_name = self.dataset_name,
        tic_name = self.tic_name)

        return train_loader, valid_loader, test_loader

    def model(self):
        model = CSDI_own(self.kwargs, self.device).to(self.device)
        return model

    def trainer(self, model, train_loader, valid_loader):
        print("Start Training")
        train(model = model, config = self.kwargs, train_loader = train_loader, valid_loader = valid_loader, foldername = self.modelfolder)
        print("Training Ends")
    
    def evaluator(self, model, test_loader):
        print("Start Evaluating")
        evaluate(model = model, test_loader = test_loader, nsample=self.nsample, scaler=1, foldername=self.modelfolder)
        print("Evaluating Ends")

    def get_quantile(self,samples,q,dim):
        return torch.quantile(samples,q,dim).cpu().numpy()
    
    def impute(self):
        data_path = "./work_dir/missing_value_imputation/data/" + self.dataset_name + "/" + self.tic_name + "_missing" + str(self.missing_ratio) + "_seed1.pk"
        result_path = "./work_dir/missing_value_imputation/save/" + self.dataset_name + "/" + self.tic_name + "_" + str(self.missing_ratio) + "/generated_outputs_nsample100.pk"
        with open(data_path, "rb") as f:
                observed_values, observed_masks, gt_masks, mean, std = pickle.load(f)

        with open(result_path, 'rb') as f:
            samples,all_target,all_evalpoint,all_observed,all_observed_time,scaler,mean_scaler = pickle.load(f)

        all_target_np = all_target.cpu().numpy()
        all_evalpoint_np = all_evalpoint.cpu().numpy()
        all_observed_np = all_observed.cpu().numpy()
        all_given_np = all_observed_np - all_evalpoint_np
        K = samples.shape[-1]
        L = samples.shape[-2]

        for k in range(K):
            samples[:,:,:,k] = samples[:,:,:,k]*std[k]+mean[k]
            all_target_np[:,:,k] = all_target_np[:,:,k]*std[k]+mean[k]

        qlist =[0.05,0.25,0.5,0.75,0.95]
        quantiles_imp= []
        for q in qlist:
            quantiles_imp.append(self.get_quantile(samples, q, dim=1)*(1-all_given_np) + all_target_np * all_given_np)

        if self.dataset_name == "dj30":
            dataset_path = "./data/portfolio_management/dj30/dj30.csv"
        elif self.dataset_name == "exchange":
            dataset_path = "./data/portfolio_management/exchange/exchange.csv"
        elif self.dataset_name == "btc":
            dataset_path = "./data/algorithmic_trading/BTC/BTC.csv"
        data = pd.read_csv(dataset_path)
        date_list = data.date.unique().tolist()
        start_date = date_list[0]
        end_date = date_list[-1]
        date_list_2 = pd.bdate_range(start=start_date, end=end_date)
        df = data.loc[data['tic']==self.tic_name]
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
        print("Missing value imputed: ", impute_list)
        save_path = './work_dir/missing_value_imputation/imputed/' + self.dataset_name
        os.makedirs(save_path, exist_ok=True)
        df_new.to_csv(save_path + "/" + self.tic_name + '.csv')
    
    def visual(self):
        data_path = "./work_dir/missing_value_imputation/data/" + self.dataset_name + "/" + self.tic_name + "_missing" + str(self.missing_ratio) + "_seed1.pk"
        result_path = "./work_dir/missing_value_imputation/save/" + self.dataset_name + "/" + self.tic_name + "_" + str(self.missing_ratio) + "/generated_outputs_nsample100.pk"
        with open(data_path, "rb") as f:
                observed_values, observed_masks, gt_masks, mean, std = pickle.load(f)

        with open(result_path, 'rb') as f:
            samples,all_target,all_evalpoint,all_observed,all_observed_time,scaler,mean_scaler = pickle.load(f)

        
        all_target_np = all_target.cpu().numpy()
        all_evalpoint_np = all_evalpoint.cpu().numpy()
        all_observed_np = all_observed.cpu().numpy()
        all_given_np = all_observed_np - all_evalpoint_np
        K = samples.shape[-1] 
        L = samples.shape[-2] 

        for k in range(K):
            samples[:,:,:,k] = samples[:,:,:,k]*std[k]+mean[k]
            all_target_np[:,:,k] = all_target_np[:,:,k]*std[k]+mean[k]

        qlist =[0.05,0.25,0.5,0.75,0.95]
        quantiles_imp= []
        for q in qlist:
            quantiles_imp.append(self.get_quantile(samples, q, dim=1)*(1-all_given_np) + all_target_np * all_given_np)


        dataind = 0 

        #plot point chart
        plt.rcParams["font.size"] = 16
        fig, axes = plt.subplots(nrows=2, ncols=5,figsize=(18, 18))
        for k in range(K):
            df = pd.DataFrame({"x":np.arange(0,L), "val":all_target_np[dataind,:,k], "y":all_evalpoint_np[dataind,:,k]})
            df = df[df.y != 0]
            df2 = pd.DataFrame({"x":np.arange(0,L), "val":all_target_np[dataind,:,k], "y":all_given_np[dataind,:,k]})
            df2 = df2[df2.y != 0]
            row = k // 5
            col = k % 5
            axes[row][col].plot(range(0,L), quantiles_imp[2][dataind,:,k], color = 'g',linestyle='solid',label='CSDI')
            axes[row][col].fill_between(range(0,L), quantiles_imp[0][dataind,:,k],quantiles_imp[4][dataind,:,k],
                            color='g', alpha=0.3)
            axes[row][col].plot(df.x,df.val, color = 'b',marker = 'o', linestyle='None')
            axes[row][col].plot(df2.x,df2.val, color = 'r',marker = 'x', linestyle='None')
            if col == 0:
                plt.setp(axes[row, 0], ylabel='value')
            plt.setp(axes[0, col], xlabel='day')
        for count in range(5):
            fig.delaxes(axes[-1][-count])
        #plt.pause(2)
        visual_path = "./work_dir/missing_value_imputation/visual/" + self.dataset_name + "/" + self.tic_name + str(self.missing_ratio)
        os.makedirs(visual_path, exist_ok=True)
        plt.savefig(visual_path + "/point_chart" + str(dataind) + ".png")
        plt.show()
        plt.close()



        if self.dataset_name == "dj30":
            dataset_path = "./data/portfolio_management/dj30/dj30.csv"
        elif self.dataset_name == "exchange":
            dataset_path = "./data/portfolio_management/exchange/exchange.csv"
        elif self.dataset_name == "btc":
            dataset_path = "./data/algorithmic_trading/BTC/BTC.csv"
        data = pd.read_csv(dataset_path)
        date_list = data.date.unique().tolist()
        start_date = date_list[0]
        end_date = date_list[-1]
        date_list_2 = pd.bdate_range(start=start_date, end=end_date)
        date = [ x.strftime('%F') for x in date_list_2]
        print("visualization date list: ", date[0:10])
        test_gt = pd.DataFrame({"datetime":np.arange(0,L)})
        test_impute = pd.DataFrame({"datetime":np.arange(0,L)})
        indicator = ['open', 'high', 'low', 'close', 'adjcp']
        for k in range(K):
            df = pd.DataFrame({"datetime":np.arange(0,L), "val":all_target_np[dataind,:,k], "y":all_evalpoint_np[dataind,:,k]})
            df = df[df.y != 0]
            df2 = pd.DataFrame({"datetime":np.arange(0,L), "val":all_target_np[dataind,:,k], "y":all_given_np[dataind,:,k]})
            df2 = df2[df2.y != 0]
            df3 = pd.concat([df,df2])
            df3 = df3.sort_values(by=["datetime"])   
            test_gt[indicator[k]] = df3['val']
            test_impute[indicator[k]] = quantiles_imp[2][dataind,:,k]
        test_gt['datetime'] = date[0:10]
        test_gt.index = pd.DatetimeIndex(test_gt['datetime'])
        test_impute['datetime'] = date[0:10]
        test_impute.index = pd.DatetimeIndex(test_impute['datetime'])

        title_name = "Candlestick for " + self.tic_name
        candlestick_path = visual_path + "/candlestick_chart" + str(dataind) + ".png"
        add_plot=[mpf.make_addplot(test_impute, ylabel="imputed price($)", type='candle', panel=1)]
        mpf.plot(test_gt,type='candle',addplot=add_plot, title=title_name, ylabel="raw price($)", style="binance", main_panel=0, panel_ratios=(1,1), savefig = candlestick_path)
        mpf.plot(test_gt,type='candle',addplot=add_plot, title=title_name, ylabel="raw price($)", style="binance", main_panel=0, panel_ratios=(1,1))

    def run(self):
        os.makedirs(self.modelfolder, exist_ok=True)
        train_loader, valid_loader, test_loader = self.data_loader()
        model = self.model()

        self.trainer(model, train_loader, valid_loader)

        self.evaluator(model, test_loader)

        self.impute()

        self.visual()





