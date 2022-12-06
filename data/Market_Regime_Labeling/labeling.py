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


class Labeling():
    def __int__(self,data,method='linear'):
        self.data_dict,self.tics=self.preprocess(data)
        if method=='linear':
            self.method='linear'
        else:
            raise Exception("Sorry, only linear model is provided for now.")
    def fit(self):
        if self.method=='linear':


    def preprocess(self,data):
        try:
            data = data['Adj Close']
            data = data.reset_index()
            data = data.rename(columns={'Date': 'date'})
        except:
            print('Please use the data formate of yfinance download data where the \'Adj Close\' and \'Date\' columns are required')
        data_dict = {}
        tics = [tic for tic in data.columns][1:]
        for tic in tics:
            temp = data.loc[:, ['date', tic]]
            temp.rename(columns={tic: 'adjcp'}, inplace=True)
            temp = temp.assign(pct_return=temp['adjcp'].pct_change().fillna(0))
            data_dict[tic] = temp
        return data_dict, tics

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
        # suppose the data is sample at 7Hz (7days / week) so fs=7 and fn=7/2, we would like to eliminate volatility on weekly scale, then we should have
        # Wn=1/(7/2)=2/7
        b, a = butter(order, Wn, btype='low', analog=False)
        y = filtfilt(b, a, data)
        return y

    def adjcp_apply_filter(self,data, Wn_adjcp, Wn_pct, order):
        data['adjcp_filtered'] = self.butter_lowpass_filter(data['adjcp'], Wn_adjcp, order)
        data['pct_return_filtered'] = self.butter_lowpass_filter(data['pct_return'], Wn_pct, order)

    def plot_lowpassfilter(self,data, name, savefig):
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
        if savefig:
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
        turning_points = find_index_of_turning(data)
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
    def linear_regession_label(self,data, y_pred_list, turning_points, low, high, normalized_coef_list):
        data = data.reset_index(drop=True)
        seg1, seg2, seg3 = sorted([low, high, 0])
        label = []
        for i in range(len(turning_points) - 1):
            x_seg = np.asarray([j for j in range(turning_points[i], turning_points[i + 1])]).reshape(-1, 1)
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

    def linear_regession_plot(self,data, tic, y_pred_list, turning_points, low, high, savefig, normalized_coef_list):
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
        if savefig:
            fig.savefig('res/linear_model/' + tic + '.png')

    def linear_regession_timewindow(self,data_ori, tic, adjcp_timewindow):
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