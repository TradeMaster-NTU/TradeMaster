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

class Labeler():
    def __init__(self,data,method='linear',parameters=['2/7','2/14','4']):
        self.preprocess(data)
        if method=='linear':
            self.method='linear'
            self.Wn_adjcp, self.Wn_pct, self.order =[float(fractions.Fraction(x)) for x in parameters]
        else:
            raise Exception("Sorry, only linear model is provided for now.")
    def fit(self):
        if self.method=='linear':
            for tic in self.tics:
                self.adjcp_apply_filter(self.data_dict[tic], self.Wn_adjcp, self.Wn_pct, self.order)
            self.turning_points_dict = {}
            self.coef_list_dict = {}
            self.norm_coef_list_dict = {}
            self.y_pred_dict = {}
            for tic in self.tics:
                coef_list, turning_points, y_pred_list, norm_coef_list = self.linear_regession_turning_points(
                    data_ori=self.data_dict[tic], tic=tic)
                self.turning_points_dict[tic] = turning_points
                self.coef_list_dict[tic] = coef_list
                self.y_pred_dict[tic] = y_pred_list
                self.norm_coef_list_dict[tic] = norm_coef_list

    def label(self,parameters=[-0.5,0.5]):
        # return a dict of label where key is the ticker and value is the label of time-series
        if self.method=='linear':
            try:
                low, high = parameters
            except:
                raise Exception(
                    "parameters shoud be [low,high] where the series would be split into 4 regimes by low,high and 0 as threshold based on slope. A value of -0.5 and 0.5 stand for -0.5% and 0.5% change per step.")
            for tic in self.tics:
                turning_points = self.turning_points_dict[tic]
                norm_coef_list = self.norm_coef_list_dict[tic]
                label = self.linear_regession_label(turning_points, low, high, norm_coef_list)
                self.data_dict[tic]['label'] = label
    def linear_regession_label(self, turning_points, low, high, normalized_coef_list):
        seg1, seg2, seg3 = sorted([low, high, 0])
        label = []
        for i in range(len(turning_points) - 1):
            coef = normalized_coef_list[i]
            if coef <= seg1:
                flag = 0
            elif coef > seg1 and coef <= seg2:
                flag = 1
            elif coef > seg2 and coef <= seg3:
                flag = 2
            elif coef > seg3:
                flag = 3
            label.extend([flag] * (turning_points[i + 1] - turning_points[i]))
        return label


    def preprocess(self,data):
        data = pd.read_csv(data)
        self.tics = data['tic'].unique()
        self.data_dict = {}
        for tic in self.tics:
            tic_data = data.loc[data['tic'] == tic, ['date','adjcp','tic']]
            tic_data.sort_values(by='date', ascending=True)
            tic_data = tic_data.assign(pct_return=tic_data['adjcp'].pct_change().fillna(0))
            self.data_dict[tic] = tic_data.reset_index(drop=True)




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

    def linear_regession_turning_points(self,data_ori, tic):
        data = data_ori.reset_index(drop=True)
        turning_points = self.find_index_of_turning(data)
        coef_list = []
        normalized_coef_list = []
        y_pred_list = []
        for i in range(len(turning_points) - 1):
            x_seg = np.asarray([j for j in range(turning_points[i], turning_points[i + 1])]).reshape(-1, 1)
            #         print(x_seg,data['adjcp_filtered'].iloc[turning_points[i]:turning_points[i+1]])
            adj_cp_model = LinearRegression().fit(x_seg,
                                                  data['adjcp_filtered'].iloc[turning_points[i]:turning_points[i + 1]])
            y_pred = adj_cp_model.predict(x_seg)
            normalized_coef_list.append(100 * adj_cp_model.coef_ / data['adjcp_filtered'].iloc[turning_points[i]])
            coef_list.append(adj_cp_model.coef_)
            y_pred_list.append(y_pred)
        return np.asarray(coef_list), np.asarray(turning_points), y_pred_list, normalized_coef_list
    def plot(self,tics,parameters):
        if self.method=='linear':
            try:
                low,high=parameters
            except:
                raise Exception("parameters shoud be [low,high] where the series would be split into 4 regimes by low,high and 0 as threshold based on slope. A value of -0.5 and 0.5 stand for -0.5% and 0.5% change per step.")
            for tic in tics:
                self.linear_regession_plot(self.data_dict[tic],tic,self.y_pred_dict[tic],self.turning_points_dict[tic],low,high,normalized_coef_list=self.norm_coef_list_dict[tic])
    def linear_regession_plot(self,data, tic, y_pred_list, turning_points, low, high, normalized_coef_list):
        data = data.reset_index(drop=True)
        fig, ax = plt.subplots(2, 1, figsize=(20, 10), constrained_layout=True)
        ax[0].plot([i for i in range(data.shape[0])], data['adjcp_filtered'])
        seg1, seg2, seg3 = sorted([low, high, 0])
        colors = list(dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS).keys())
        for i in range(len(turning_points) - 1):
            x_seg = np.asarray([j for j in range(turning_points[i], turning_points[i + 1])]).reshape(-1, 1)
            y_pred = y_pred_list[i]
            coef = normalized_coef_list[i]
            if coef <= seg1:
                flag = 0
            elif coef > seg1 and coef <= seg2:
                flag = 1
            elif coef > seg2 and coef <= seg3:
                flag = 2
            elif coef > seg3:
                flag = 3
            ax[1].plot(x_seg, y_pred, color=colors[flag], label='cat' + str(flag))
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())
        ax[0].set_title(tic + '_linear_regression_regime', fontsize=20)
        if not os.path.exists('res/linear_model/'):
            os.makedirs('res/linear_model/')
        fig.savefig('res/linear_model/' + tic + '.png')

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