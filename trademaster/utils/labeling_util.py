import pandas as pd
import yfinance as yf
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

import statsmodels.api as sm
from scipy.signal import butter,filtfilt
from matplotlib import colors as mcolors
from sklearn.linear_model import LinearRegression
from random import sample
import fractions
import os
from sklearn.manifold import TSNE
from tslearn.clustering import TimeSeriesKMeans
from tslearn.utils import to_time_series_dataset
import pickle
import re
import matplotlib.font_manager as font_manager

class Labeler():
    def __init__(self,data,method='linear',parameters=['2/7','2/14','4']):
        plt.ioff()
        self.preprocess(data)
        if method=='linear':
            self.method='linear'
            self.Wn_adjcp, self.Wn_pct, self.order =[float(fractions.Fraction(x)) for x in parameters]
        else:
            raise Exception("Sorry, only linear model is provided for now.")
    def fit(self,regime_number,length_limit):
        if self.method=='linear':
            for tic in self.tics:
                self.adjcp_apply_filter(self.data_dict[tic], self.Wn_adjcp, self.Wn_pct, self.order)
            self.turning_points_dict = {}
            self.coef_list_dict = {}
            self.norm_coef_list_dict = {}
            self.y_pred_dict = {}
            self.regime_number=regime_number
            self.length_limit=length_limit
            for tic in self.tics:
                coef_list, turning_points, y_pred_list, norm_coef_list = self.linear_regession_turning_points(
                    data_ori=self.data_dict[tic], tic=tic,length_constrain=self.length_limit)
                self.turning_points_dict[tic] = turning_points
                self.coef_list_dict[tic] = coef_list
                self.y_pred_dict[tic] = y_pred_list
                self.norm_coef_list_dict[tic] = norm_coef_list

    def label(self,parameters,work_dir=os.getcwd()):
        # return a dict of label where key is the ticker and value is the label of time-series
        if self.method=='linear':
            try:
                low, high = parameters
            except:
                raise Exception(
                    "parameters shoud be [low,high] where the series would be split into 4 regimes by low,high and 0 as threshold based on slope. A value of -0.5 and 0.5 stand for -0.5% and 0.5% change per step.")
            self.all_data_seg = []
            self.all_label_seg = []
            self.all_index_seg = []
            for tic in self.tics:
                turning_points = self.turning_points_dict[tic]
                norm_coef_list = self.norm_coef_list_dict[tic]
                label,data_seg,label_seg,index_seg = self.linear_regession_label(self.data_dict[tic],turning_points, low, high, norm_coef_list,tic,self.regime_number)
                self.data_dict[tic]['label'] = label
                self.all_data_seg.extend(data_seg)
                self.all_label_seg.extend(label_seg)
                self.all_index_seg.extend(index_seg)
            interpolated_pct_return_data_seg = np.array(self.interpolation(self.all_data_seg))
            try:
              self.TSNE_run(interpolated_pct_return_data_seg)
            except:
              print('not able to do TSNE') 
            try:
              self.stock_DWT(work_dir)
            except:
              print('not able to do clustering') 

    def linear_regession_label(self,data, turning_points, low, high, normalized_coef_list, tic,
                               regime_num=4):
        data = data.reset_index(drop=True)['pct_return_filtered']
        data_seg = []
        seg1, seg2, seg3 = sorted([low, high, 0])
        label = []
        label_seg = []
        index_seg = []
        for i in range(len(turning_points) - 1):
            coef = normalized_coef_list[i]
            flag = self.regime_flag(regime_num, coef, [seg1, seg2, seg3])
            label.extend([flag] * (turning_points[i + 1] - turning_points[i]))
            if turning_points[i + 1] - turning_points[i] > 2:
                data_seg.append(data.iloc[turning_points[i]:turning_points[i + 1]].to_list())
                label_seg.append(flag)
                index_seg.append(tic + '_' + str(i))
        return label, data_seg, label_seg, index_seg

    def regime_flag(self,regime_num, coef, parameters):
        seg1, seg2, seg3 = parameters
        if regime_num == 4:
            if coef <= seg1:
                flag = 0
            elif coef > seg1 and coef <= seg2:
                flag = 1
            elif coef > seg2 and coef <= seg3:
                flag = 2
            elif coef > seg3:
                flag = 3
        elif regime_num == 3:
            if coef <= seg1:
                flag = 0
            elif coef > seg1 and coef <= seg3:
                flag = 1
            elif coef > seg3:
                flag = 2
        else:
            raise Exception('This regime num is currently not supported')
        return flag
    def preprocess(self,data):
        data = pd.read_csv(data)
        self.tics = data['tic'].unique()
        self.data_dict = {}
        for tic in self.tics:
            try:
                tic_data = data.loc[data['tic'] == tic, ['date','tic','open','high','low','close','adjcp']]
            except:
                tic_data = data.loc[data['tic'] == tic, ['date', 'tic', 'adjcp']]
            tic_data.sort_values(by='date', ascending=True)
            tic_data = tic_data.assign(pct_return=tic_data['adjcp'].pct_change().fillna(0))
            self.data_dict[tic] = tic_data.reset_index(drop=True)

    def stock_DWT(self,work_dir):
        data_by_tic = []
        data_by_tic_1 = []
        for tic in self.tics:
            try:
                data_by_tic.append(self.data_dict[tic].loc[:, ['open', 'high', 'low', 'close', 'adjcp', 'pct_return']].values)
            except:
                data_by_tic.append(
                    self.data_dict[tic].loc[:, ['adjcp', 'pct_return']].values)
            data_by_tic_1.append(self.data_dict[tic].loc[:, 'pct_return'].values)
        fitting_data = to_time_series_dataset(data_by_tic)
        fitting_data_1=to_time_series_dataset(data_by_tic_1)
        km_stock = TimeSeriesKMeans(n_clusters=6, metric="dtw", max_iter=50, max_iter_barycenter=100, n_jobs=50,
                                    verbose=0).fit(fitting_data)
        label_stock = km_stock.predict(fitting_data)
        output = open(os.path.join(work_dir,'DWT_stocks.pkl'), 'wb')
        pickle.dump(km_stock, output, -1)
        output.close()
        output = open(os.path.join(work_dir,'DWT_label_stocks.pkl'), 'wb')
        pickle.dump(label_stock, output, -1)
        output.close()
        for i in range(len(self.tics)):
            self.data_dict[self.tics[i]]['stock_type']=label_stock[i]
        tsne_model = TSNE(n_components=3, perplexity=25, n_iter=300)
        tsne_results = tsne_model.fit_transform(fitting_data_1.reshape(fitting_data_1.shape[0],fitting_data_1.shape[1]))
        self.TSNE_plot(tsne_results,label_stock,'_stock_cluster',folder_name=self.plot_path)

    def plot_ori(self,data, name):
        fig, ax = plt.subplots(1, 1, figsize=(20, 10), constrained_layout=True)
        if isinstance(data['date'][0], str):
            date = data['date'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))
        else:
            date = data['date']
        ax.plot(date, data['adjcp'])
        ax.xaxis.set_major_locator(mdates.YearLocator(base=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%M'))
        ax.set_title(name + '_adjcp', fontsize=20)
        ax.grid(True)
        if not os.path.exists('res/'):
            os.makedirs('res/')
        fig.savefig('res/' + name + '_adjcp' + '.png')

    def plot_pct(self,data, name):
        fig, ax = plt.subplots(1, 1, figsize=(20, 10), constrained_layout=True)
        if isinstance(data['date'][0], str):
            date = data['date'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))
        else:
            date = data['date']
        ax.plot(date, data['pct_return'])
        ax.xaxis.set_major_locator(mdates.YearLocator(base=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%M'))
        ax.set_title(name + 'pct_return', fontsize=20)
        ax.grid(True)
        if not os.path.exists('res/'):
            os.makedirs('res/')
        fig.savefig('res/' + name + '_pct_return' + '.png')

    def plot_both(self,data, name):
        self.plot_ori(data, name)
        self.plot_pct(data, name)

    def plot_filter(self,data, name, low=6, high=32, K=12):
        # see sm.tsa.filters.bkfilter for more detail, this method is not applied to the pipline for now
        filtered_data = sm.tsa.filters.bkfilter(data[['adjcp', 'pct_return']], low, high, K)
        if isinstance(data['date'][0], str):
            date = data['date'][K:-K].apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))
        else:
            date = data['date'][K:-K]
        fig, ax = plt.subplots(2, 1, figsize=(20, 10), constrained_layout=True)
        ax[0].plot(date, filtered_data['adjcp_cycle'], label='adjcp_cycle')
        ax[1].plot(date, filtered_data['pct_return_cycle'], label='pct_return_cycle')
        ax[0].xaxis.set_major_locator(mdates.YearLocator(base=1))
        ax[0].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%M'))
        ax[1].xaxis.set_major_locator(mdates.YearLocator(base=1))
        ax[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%M'))
        ax[0].set_title(name + '_adjcp_cycle', fontsize=20)
        ax[1].set_title(name + '_pct_return_cycle', fontsize=20)
        ax[0].grid(True)
        ax[1].grid(True)

    def butter_lowpass_filter(self,data, Wn, order):
        # It is strongly recommended to adjust the Wn based on different data.
        # You can refer to https://en.wikipedia.org/wiki/Butterworth_filter for parameter setting
        # suppose the data is sample at 7Hz (7days / week) so fs=7 and fn=7/2, we would like to eliminate volatility on weekly scale, then we should have
        # Wn=1/(7/2)=2/7
        b, a = butter(order, Wn, btype='low', analog=False)
        y = filtfilt(b, a, data)
        return y

    def adjcp_apply_filter(self,data, Wn_adjcp, Wn_pct, order):
        data['adjcp_filtered'] = self.butter_lowpass_filter(data['adjcp'], Wn_adjcp, order)
        data['pct_return_filtered'] = self.butter_lowpass_filter(data['pct_return'], Wn_pct, order)

    def plot_lowpassfilter(self,data, name):
        fig, ax = plt.subplots(2, 1, figsize=(20, 10), constrained_layout=True)
        if isinstance(data['date'][0], str):
            date = data['date'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))
        else:
            date = data['date']
        ax[0].plot(date, data['adjcp_filtered'])
        ax[0].xaxis.set_major_locator(mdates.YearLocator(base=1))
        ax[0].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%M'))
        ax[0].set_title(name + '_adjcp_filtered', fontsize=20)
        ax[1].plot(date, data['pct_return_filtered'])
        ax[1].xaxis.set_major_locator(mdates.YearLocator(base=1))
        ax[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%M'))
        ax[1].set_title(name + '_pct_return_filtered', fontsize=20)
        if not os.path.exists('res/'):
            os.makedirs('res/')
        fig.savefig('res/' + name + 'filtered' + '.png')

    def find_index_of_turning(self,data):
        turning_points = [0]
        data = data.reset_index(drop=True)
        for i in range(data['pct_return_filtered'].size - 1):
            if data['pct_return_filtered'][i] * data['pct_return_filtered'][i + 1] < 0:
                turning_points.append(i + 1)
        if turning_points[-1] != data['pct_return_filtered'].size:
            turning_points.append(data['pct_return_filtered'].size)
        return turning_points

    def linear_regession_turning_points(self,data_ori, tic,length_constrain=0):
        data = data_ori.reset_index(drop=True)
        turning_points = self.find_index_of_turning(data)
        turning_points_new = [turning_points[0]]
        if length_constrain != 0:
            for i in range(1, len(turning_points) - 1):
                if turning_points[i] - turning_points_new[-1] >= length_constrain:
                    turning_points_new.append(turning_points[i])
            turning_points_new.append(turning_points[-1])
            turning_points = turning_points_new
        coef_list = []
        normalized_coef_list = []
        y_pred_list = []
        for i in range(len(turning_points) - 1):
            x_seg = np.asarray([j for j in range(turning_points[i], turning_points[i + 1])]).reshape(-1, 1)
            adj_cp_model = LinearRegression().fit(x_seg,
                                                  data['adjcp_filtered'].iloc[turning_points[i]:turning_points[i + 1]])
            y_pred = adj_cp_model.predict(x_seg)
            normalized_coef_list.append(100 * adj_cp_model.coef_ / data['adjcp_filtered'].iloc[turning_points[i]])
            coef_list.append(adj_cp_model.coef_)
            y_pred_list.append(y_pred)
        return np.asarray(coef_list), np.asarray(turning_points), y_pred_list, normalized_coef_list

    def plot(self,tics,parameters,data_path,model_id):
        self.plot_path =os.path.join(os.path.dirname(os.path.realpath(data_path)),'MDM_linear',model_id)
        if not os.path.exists(self.plot_path):
            os.makedirs(self.plot_path)
        if self.method=='linear':
            try:
                low,high=parameters
            except:
                raise Exception("parameters shoud be [low,high] where the series would be split into 4 regimes by low,high and 0 as threshold based on slope. A value of -0.5 and 0.5 stand for -0.5% and 0.5% change per step.")
            for tic in tics:
                paths=[]
                paths.append(self.linear_regession_plot(self.data_dict[tic],tic,self.y_pred_dict[tic],self.turning_points_dict[tic],low,high,normalized_coef_list=self.norm_coef_list_dict[tic],folder_name=self.plot_path))
                return paths
            try:
              self.TSNE_plot(self.tsne_results,self.all_label_seg,folder_name=self.plot_path)
            except:
              print('not able to plot TSNE')
    def linear_regession_plot(self,data, tic, y_pred_list, turning_points, low, high, normalized_coef_list,folder_name=None):
        data = data.reset_index(drop=True)
        fig, ax = plt.subplots(1, 1, figsize=(20, 10), constrained_layout=True)
        seg1, seg2, seg3 = sorted([low, high, 0])
        colors = list(dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS).keys())
        for i in range(len(turning_points) - 1):
            x_seg = np.asarray([j for j in range(turning_points[i], turning_points[i + 1])]).reshape(-1, 1)
            y_pred = y_pred_list[i]
            coef = normalized_coef_list[i]
            flag=self.regime_flag(self.regime_number,coef,[seg1, seg2, seg3])
            ax.plot(x_seg,data['adjcp'].iloc[turning_points[i]:turning_points[i + 1]], color=colors[flag], label='market style ' + str(flag))
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        font = font_manager.FontProperties(weight='bold',
                                           style='normal', size=16)
        plt.legend(by_label.values(), by_label.keys(), prop=font)
        ax.set_title(tic + '_linear_regression_regime', fontsize=20)
        plot_path=folder_name
        if not os.path.exists(plot_path):
            os.makedirs(plot_path)
        fig_path=plot_path+'_'+tic+'.png'
        fig.savefig(fig_path)
        plt.close(fig)
        return os.path.abspath(fig_path).replace("\\", "/")

    def linear_regession_timewindow(self,data_ori, tic, adjcp_timewindow):
        # This is the version of linear regession that does not use a turning point to segment. Instead, it applys a fixed-length time winodw.
        # This can be helpful to process data that is extremely volatile, or you simply want a long-term regime. However you can achieve similar result by applying stronger filter.
        data = data_ori.iloc[:adjcp_timewindow * (data_ori['adjcp_filtered'].size // adjcp_timewindow), :]
        adjcp_window_data = [
            data[['adjcp_filtered']][i * adjcp_timewindow:(i + 1) * adjcp_timewindow].to_numpy().reshape(-1) for i in
            range(data['adjcp_filtered'].size // adjcp_timewindow)]
        coef_list = []
        fig, ax = plt.subplots(2, 1, figsize=(20, 10), constrained_layout=True)
        ax[0].plot([i for i in range(data.shape[0])], data['adjcp_filtered'])
        for i, data_seg in enumerate(adjcp_window_data):
            x_seg = np.asarray([i * adjcp_timewindow + j for j in range(adjcp_timewindow)]).reshape(-1, 1)
            adj_cp_model = LinearRegression().fit(x_seg, data_seg)
            y_pred = adj_cp_model.predict(x_seg)
            ax[1].plot([i * adjcp_timewindow + j for j in range(adjcp_timewindow)], y_pred)
            coef_list.append(adj_cp_model.coef_)
        return coef_list

    def interpolation(self,data):
        max_len = max([len(d) for d in data])
        for i, d in enumerate(data):
            l = len(d)
            to_fill = max_len - l
            if to_fill != 0:
                interval = max_len // to_fill
                for j in range(to_fill):
                    idx = (interval + 1) * j + interval
                    data[i].insert(min(idx, len(data[i]) - 1), float('nan'))
            data[i] = pd.Series(data[i]).interpolate(method='polynomial', order=2)
        return data

    def TSNE_run(self,data_seg):
        interpolated_pct_return_data_seg = np.array(self.interpolation(data_seg))
        self.tsne = TSNE(n_components=2,perplexity=40, n_iter=300)
        self.tsne_results = self.tsne.fit_transform(interpolated_pct_return_data_seg)

    def TSNE_plot(self,data, label_list,title='',folder_name=None):
        colors = list(dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS).keys())
        fig, ax = plt.subplots(1, 1, figsize=(20, 10), constrained_layout=True)
        for i in range(len(data) - 1):
            label = label_list[i]
            ax.scatter(data[i][0], data[i][1], color=colors[label], alpha=0.2, label='cluster' + str(label))
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())
        plt.title('TSNE', fontsize=20)
        plot_path = folder_name
        if not os.path.exists(plot_path):
            os.makedirs(plot_path)
        fig.savefig(plot_path+'TSNE'+title+'.png')
        plt.close(fig)

