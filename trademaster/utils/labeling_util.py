import math
import pandas as pd
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import statsmodels.api as sm
from scipy.signal import butter, filtfilt
from matplotlib import colors as mcolors
from sklearn.linear_model import LinearRegression
import os
from sklearn.manifold import TSNE
from tslearn.clustering import TimeSeriesKMeans
from tslearn.utils import to_time_series_dataset
import pickle
import matplotlib.font_manager as font_manager
from tqdm import tqdm
from fastdtw import fastdtw


class Dynamic_labeler():
    def __init__(self, labeling_method, dynamic_num, low, high, normalized_coef_list, data, turning_points):
        self.labeling_method = labeling_method
        self.dynamic_num = dynamic_num
        if self.labeling_method == 'slope':
            low, _, high = sorted([low, high, 0])
            self.segments = []
            if high!=low:
                for i in range(1, self.dynamic_num):
                    self.segments.append(low + (high - low) / (dynamic_num-2) * i)
            else:
                self.segments.append(low)
        elif self.labeling_method == 'quantile':
            self.segments = []
            # find the quantile of normalized_coef_list
            for i in range(1, self.dynamic_num):
                self.segments.append(np.quantile(normalized_coef_list, i / dynamic_num))
        elif self.labeling_method == 'DTW':
            # segment the data by turning points
            self.segments = []
            for i in range(len(turning_points) - 1):
                self.segments.append(data['pct_return_filtered'][turning_points[i]:turning_points[i + 1]])
            # run the DTW algorithm to cluster the segments into dynamic_num clusters
            self.labels = self.DTW_clustering(self.segments)
        else:
            raise Exception("Sorry, only slope,quantile and DTW labeling_method are provided for now.")

    def DTW_clustering(self, data):
        fitting_data = to_time_series_dataset(data)
        km_stock = TimeSeriesKMeans(n_clusters=self.dynamic_num, metric="dtw", max_iter=100, max_iter_barycenter=100,
                                    n_jobs=50,
                                    verbose=0).fit(fitting_data)
        labels = km_stock.predict(fitting_data)
        return labels

    def get(self, coef):
        if self.labeling_method == 'DTW':
            return self.labels[coef]
        elif self.labeling_method == 'slope' or self.labeling_method == 'quantile':
            # find the place where coef falls into in segments
            for i in range(self.dynamic_num - 1):
                if coef <= self.segments[i]:
                    flag = i
                    return flag
            return self.dynamic_num - 1


class Worker():
    def __init__(self, data_path, method='slice_and_merge', filter_strength=1, key_indicator='adjcp', timestamp='date', tic='tic',
                 labeling_method='slope', min_length_limit=-1, merging_threshold=-1, merging_metric='DTW_distance',merging_dynamic_constraint=-1):
        plt.ioff()
        self.key_indicator = key_indicator
        self.timestamp = timestamp
        self.tic = tic
        self.labeling_method = labeling_method
        # the hard length limit is the hard constraint of the minium ticks of a continuous segment, which means that any volatility
        # with length less than the hard length limit will be considered as noise.
        self.min_length_limit = min_length_limit
        self.merging_metric = merging_metric
        self.merging_threshold = merging_threshold
        if merging_dynamic_constraint < 0:
            self.merging_dynamic_constraint = float('inf')
        else:
            self.merging_dynamic_constraint = merging_dynamic_constraint
        self.do_TSNE=False
        self.do_DTW=False
        if method == 'slice_and_merge':
            self.method = 'slice_and_merge'
            # calculate the parameters for filtering
            self.order = 4
            self.Wn_key_indicator = self.filter_parameters_calculation(filter_strength)
        else:
            raise Exception("Sorry, only linear model is provided for now.")
        self.preprocess(data_path)

    def filter_parameters_calculation(self, filter_strength):
        if self.min_length_limit != -1:
            filter_period = self.min_length_limit
        else:
            filter_period = 7  # default filter period

        # use the min_length_limit to calculate the Wn_key_indicator and Wn_pct
        # the max Wn_key_indicator is 2, and the max Wn_pct is 2 for not filtering
        Wn_key_indicator = min(2 / (filter_period * filter_strength), 2)
        # Wn_pct=min(2,2/(filter_period*Wn_pct_factor))
        return Wn_key_indicator

    def fit(self, dynamic_number, max_length_expectation, min_length_limit):
        if self.method == 'slice_and_merge':
            self.turning_points_dict = {}
            self.coef_list_dict = {}
            self.norm_coef_list_dict = {}
            self.y_pred_dict = {}
            self.dynamic_num = dynamic_number
            self.max_length_expectation = max_length_expectation
            self.min_length_limit = min_length_limit
            for i,tic in enumerate(self.tics):
                coef_list, turning_points, y_pred_list, norm_coef_list = self.get_turning_points(
                    data_ori=self.data_dict[tic], tic=tic, max_length_expectation=self.max_length_expectation)
                print('finish fitting ' + tic,' Total process:',i, '/', len(self.tics))
                self.turning_points_dict[tic] = turning_points
                self.coef_list_dict[tic] = coef_list
                self.y_pred_dict[tic] = y_pred_list
                self.norm_coef_list_dict[tic] = norm_coef_list


    def label(self, parameters, work_dir=os.getcwd()):
        # return a dict of label where key is the ticker and value is the label of time-series
        if self.method == 'slice_and_merge':
            try:
                low, high = parameters
            except:
                raise Exception(
                    "parameters shoud be [low,high] where the series would be split into 4 dynamics by low,high and 0 as threshold based on slope. A value of -0.5 and 0.5 stand for -0.5% and 0.5% change per step.")
            self.all_data_seg = []
            self.all_label_seg = []
            self.all_index_seg = []
            for tic in self.tics:
                turning_points = self.turning_points_dict[tic]
                norm_coef_list = self.norm_coef_list_dict[tic]
                if self.labeling_method=='slope' and self.dynamic_num==2:
                    print('For labeling_method==slope, and dynamic number==2 we will take the mean of the low and high as the seperation slope')
                    low = (low+high)/2
                    high = low
                elif low>=high:
                    # auto zooming low high according to the coef_list
                    # if dynamic number is 4, then the low and high should be the 25% and 75% of the coef_list
                    # uf dynamic number is 3, then the low and high should be the 33% and 66% of the coef_list
                    # if dynamic number is 2, then the low and high should be the 50% and 50% of the coef_list
                    # the following code is to calculate the low and high
                    coef_list = norm_coef_list
                    coef_list = sorted(coef_list)
                    high = coef_list[int(((self.dynamic_num - 1) / self.dynamic_num) * len(coef_list))][0]
                    low = coef_list[int((1 / self.dynamic_num) * len(coef_list))][0]
                    print(f' tic {tic} auto set low and high, will be used if labeling method is slope, low: ', low, ' high: ', high)

                label, data_seg, label_seg, index_seg = self.get_label(self.data_dict[tic], turning_points,
                                                                       low, high, norm_coef_list, tic,
                                                                       self.dynamic_num,labeling_method=self.labeling_method)
                self.data_dict[tic]['label'] = label
                self.all_data_seg.extend(data_seg)
                self.all_label_seg.extend(label_seg)
                self.all_index_seg.extend(index_seg)
            if self.do_TSNE:
                interpolated_pct_return_data_seg = np.array(self.interpolation(self.all_data_seg))
                try:
                  print('doing TSNE on data for visualization')
                  self.TSNE_run(interpolated_pct_return_data_seg)
                except:
                  print('not able to do TSNE')
            if len(self.tics) > 1 and self.do_DTW:
                try:
                    print('Doing DTW clustering of multiple tics')
                    self.tic_DTW(work_dir)
                except:
                    print('not able to do clustering')

    def get_label(self, data, turning_points, low, high, normalized_coef_list, tic,
                  dynamic_num=4,labeling_method=None):
        data = data.reset_index(drop=True)
        data_seg = []

        label = []
        label_seg = []
        index_seg = []
        self.dynamic_flag = Dynamic_labeler(labeling_method=labeling_method, dynamic_num=dynamic_num, low=low,
                                            high=high,
                                            normalized_coef_list=normalized_coef_list, data=data,
                                            turning_points=turning_points)
        data = data['pct_return_filtered']
        for i in range(len(turning_points) - 1):
            if labeling_method == 'slope' or labeling_method == 'quantile':
                coef = normalized_coef_list[i]
            elif labeling_method == 'DTW':
                coef = i
            flag = self.dynamic_flag.get(coef)
            label.extend([flag] * (turning_points[i + 1] - turning_points[i]))
            if turning_points[i + 1] - turning_points[i] > 2:
                data_seg.append(data.iloc[turning_points[i]:turning_points[i + 1]].to_list())
                label_seg.append(flag)
                index_seg.append(tic + '_' + str(i))
        return label, data_seg, label_seg, index_seg

    def preprocess(self, data):
        # parse the extention of the data file
        if data.split('.')[-1] == 'csv':
            data = pd.read_csv(data)
        elif data.split('.')[-1] == 'feather':
            data = pd.read_feather(data)
        # assign tic if not exist
        if self.tic not in data.columns:
            data[self.tic] = self.tic
        self.tics = data[self.tic].unique()
        self.data_dict = {}

        for tic in self.tics:
            tic_data = data.loc[data[self.tic] == tic, [self.timestamp, self.tic, self.key_indicator]]
            tic_data.sort_values(by=self.timestamp, ascending=True)
            tic_data = tic_data.assign(pct_return=tic_data[self.key_indicator].pct_change().fillna(0))
            self.data_dict[tic] = tic_data.reset_index(drop=True)
        for tic in self.tics:
            self.adjcp_apply_filter(self.data_dict[tic], self.Wn_key_indicator, self.order)
            self.data_dict[tic] = self.data_dict[tic].assign(
                pct_return_filtered=self.data_dict[tic]['key_indicator_filtered'].pct_change().fillna(0))

    def tic_DTW(self, work_dir):
        data_by_tic = []
        data_by_tic_1 = []
        for tic in self.tics:
            data_by_tic.append(
                self.data_dict[tic].loc[:, ['pct_return']].values)
            # data_by_tic_1.append(self.data_dict[tic].loc[:, 'pct_return'].values)
        fitting_data = to_time_series_dataset(data_by_tic)
        # fitting_data_1 = to_time_series_dataset(data_by_tic_1)
        km_stock = TimeSeriesKMeans(n_clusters=5, metric="dtw", max_iter=50, max_iter_barycenter=100, n_jobs=50,
                                    verbose=0).fit(fitting_data)
        tic_label = km_stock.predict(fitting_data)
        output = open(os.path.join(work_dir, 'DTW_tics.pkl'), 'wb')
        pickle.dump(km_stock, output, -1)
        output.close()
        output = open(os.path.join(work_dir, 'DTW_tics_label.pkl'), 'wb')
        pickle.dump(tic_label, output, -1)
        output.close()
        for i in range(len(self.tics)):
            self.data_dict[self.tics[i]]['tic_label'] = tic_label[i]
        # tsne_model = TSNE(n_components=3, perplexity=25, n_iter=300)
        # tsne_results = tsne_model.fit_transform(
        #     fitting_data_1.reshape(fitting_data_1.shape[0], fitting_data_1.shape[1]))
        # self.TSNE_plot(tsne_results, tic_label, '_tic_cluster', folder_name=self.plot_path)

    def plot_indicator(self, data, name):
        fig, ax = plt.subplots(1, 1, figsize=(20, 10), constrained_layout=True)
        if isinstance(data[self.timestamp][0], str):
            date = data[self.timestamp].apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))
        else:
            date = data[self.timestamp]
        ax.plot(date, data[self.key_indicator])
        ax.xaxis.set_major_locator(mdates.YearLocator(base=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%M'))
        ax.set_title(name + '_indicator', fontsize=20)
        ax.grid(True)
        if not os.path.exists('res/'):
            os.makedirs('res/')
        fig.savefig('res/' + name + '_indicator' + '.png')

    def plot_pct(self, data, name):
        fig, ax = plt.subplots(1, 1, figsize=(20, 10), constrained_layout=True)
        if isinstance(data[self.timestamp][0], str):
            date = data[self.timestamp].apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))
        else:
            date = data[self.timestamp]
        ax.plot(date, data['pct_return'])
        ax.xaxis.set_major_locator(mdates.YearLocator(base=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%M'))
        ax.set_title(name + 'pct_return', fontsize=20)
        ax.grid(True)
        if not os.path.exists('res/'):
            os.makedirs('res/')
        fig.savefig('res/' + name + '_pct_return' + '.png')

    def plot_both(self, data, name):
        self.plot_indicator(data, name)
        self.plot_pct(data, name)

    def plot_filter(self, data, name, low=6, high=32, K=12):
        # see sm.tsa.filters.bkfilter for more detail, this method is not applied to the pipline for now
        filtered_data = sm.tsa.filters.bkfilter(data[[self.key_indicator, 'pct_return']], low, high, K)
        if isinstance(data[self.timestamp][0], str):
            date = data[self.timestamp][K:-K].apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))
        else:
            date = data[self.timestamp][K:-K]
        fig, ax = plt.subplots(2, 1, figsize=(20, 10), constrained_layout=True)
        ax[0].plot(date, filtered_data['adjcp_cycle'], label='adjcp_cycle')
        ax[1].plot(date, filtered_data['pct_return_cycle'], label='pct_return_cycle')
        ax[0].xaxis.set_major_locator(mdates.YearLocator(base=1))
        ax[0].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%M'))
        ax[1].xaxis.set_major_locator(mdates.YearLocator(base=1))
        ax[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%M'))
        ax[0].set_title(name + '_indicator_cycle', fontsize=20)
        ax[1].set_title(name + '_pct_return_cycle', fontsize=20)
        ax[0].grid(True)
        ax[1].grid(True)

    def butter_lowpass_filter(self, data, Wn, order):
        # It is strongly recommended to adjust the Wn based on different data.
        # You can refer to https://en.wikipedia.org/wiki/Butterworth_filter for parameter setting
        # suppose the data is sample at 7Hz (7days / week) so fs=7 and fn=7/2, we would like to eliminate volatility on weekly scale, then we should have
        # Wn=1/(7/2)=2/7
        b, a = butter(order, Wn, btype='low', analog=False)
        y = filtfilt(b, a, data)
        return y

    def adjcp_apply_filter(self, data, Wn_indicator, order):
        data['key_indicator_filtered'] = self.butter_lowpass_filter(data[self.key_indicator], Wn_indicator, order)
        # print(self.key_indicator)
        # print(data[self.key_indicator])
        # print(data['key_indicator_filtered'])
        # plot the filtered data and save it to res folder
        # self.plot_lowpassfilter(data, 'filter_test')
        # print(data[['key_indicator_filtered', 'pct_return_filtered']])

    def plot_lowpassfilter(self, data, name):
        fig, ax = plt.subplots(2, 1, figsize=(20, 10), constrained_layout=True)
        if isinstance(data[self.timestamp][0], str):
            date = data[self.timestamp].apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))
        else:
            date = data[self.timestamp]
        ax[0].plot(date, data['key_indicator_filtered'])
        ax[0].xaxis.set_major_locator(mdates.YearLocator(base=1))
        ax[0].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%M'))
        ax[0].set_title(name + '_indicator_filtered', fontsize=20)
        ax[1].plot(date, data['pct_return_filtered'])
        ax[1].xaxis.set_major_locator(mdates.YearLocator(base=1))
        ax[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%M'))
        ax[1].set_title(name + '_pct_return_filtered', fontsize=20)
        if not os.path.exists('res/'):
            os.makedirs('res/')
        fig.savefig('res/' + name + 'filtered' + '.png')

    def find_index_of_turning(self, data):
        turning_points = [0]
        data = data.reset_index(drop=True)
        for i in range(data['pct_return_filtered'].size - 1):
            if data['pct_return_filtered'][i] * data['pct_return_filtered'][i + 1] < 0:
                turning_points.append(i + 1)
        if turning_points[-1] != data['pct_return_filtered'].size:
            turning_points.append(data['pct_return_filtered'].size)
        # the last turning point is the end of the data
        return turning_points

    def get_mdd(self, seg):
        # get max drawdown of a segment
        mdd = 0
        peak = seg[0]
        for value in seg:
            if value > peak:
                peak = value
            dd = (peak - value) / peak
            if dd > mdd:
                mdd = dd
        return mdd

    def calculate_distance(self, seg1, seg2, iteration_count, labeling_method='default'):
        # calculate the distance between two segments
        if labeling_method == 'default':
            labeling_method = self.merging_metric
        if labeling_method == 'DTW_distance':
            # the sampling time increase as the iteration_count increase
            distance = self.calculate_dtw_distance(seg1, seg2, iteration_count * 3 + 10)
        return distance

    def calculate_dtw_distance(self, seg1, seg2, max_sample_number=3):
        # calculate the dynamic time warping distance between two segments
        # roll the shorter segment on the longer one with step_size, and calculate the mean distance

        # decide the step size and slice length based on the max_calulation_number
        # we want to include every point in the longer segment at least once/ the longer segment slice length is comparable to the shorter segment/ step size is not too small

        if len(seg1) > len(seg2):
            longer = seg1
            shorter = seg2
        else:
            longer = seg2
            shorter = seg1

        step_size = max(1, math.floor((len(longer) - len(shorter)) / max_sample_number))
        # slice_length=int(len(longer)/max_sample_number)
        slice_length = len(shorter)

        distances = []
        for i in range(0, len(longer) - len(shorter), step_size):
            distance, paths = fastdtw(shorter, longer[i:i + slice_length])
            distances.append(distance)
        # normalize the distance by the length of the shorter segment and mean value of the shorter segment
        # print(np.mean(distances),(slice_length * np.mean(shorter)))
        return np.mean(distances) / (slice_length * np.mean(shorter))

    def get_turning_points(self, data_ori, tic, max_length_expectation=0):
        """
        1. segment the data into chunks based on turning points(where all neighbors have the opposite slope)
        2. if the length is smaller than min_length_limit, merge the chunk with its neighbor
        3. Calculate the slope
        4. While the chunk does not satisfy the length limit, and metric satisfied the merging_threshold:
             1.merge the chunk with its neighbor
             2.recalculate the slope
        if the self.merging_dynamic_constraint is not -1, we would label the segment every time before merging(except the first time) /
        and prohibit the merging if the distance of the label if larger than the merging_dynamic_constraint (the labeling method is 'quantile')
        """


        data = data_ori.reset_index(drop=True)

        # 1. segment the data into chunks based on turning points(where all neighbors have the opposite slope)
        turning_points = self.find_index_of_turning(data)
        # make every element in turning_points as a list
        turning_points = [[i] for i in turning_points]
        turning_points_new = [[turning_points[0][0]]]

        # 2.if the length is smaller than min_length_limit, merge the chunk with its neighbor


        for i in range(1, len(turning_points) - 1):
            if turning_points[-1][0] - turning_points[i][0] -1 < self.min_length_limit:
                # there is no enough tics on the right side merge them all to the last segment
                for j in range(i, len(turning_points)-1):
                    turning_points_new[-1].extend(turning_points[j])
                break
            elif turning_points[i][0] - turning_points_new[-1][0] >= self.min_length_limit:
                # no need to merge
                turning_points_new.append(turning_points[i])
            else:
                # merge this point into the current segment
                turning_points_new[-1].extend(turning_points[i])
        turning_points_new.append(turning_points[-1])
        turning_points = turning_points_new

        # 2. Calculate the slope
        coef_list = []
        normalized_coef_list = []
        y_pred_list = []
        for i in range(len(turning_points) - 1):
            x_seg = np.asarray([j for j in range(turning_points[i][0], turning_points[i + 1][0])]).reshape(-1, 1)
            adj_cp_model = LinearRegression().fit(x_seg,
                                                  data['key_indicator_filtered'].iloc[
                                                  turning_points[i][0]:turning_points[i + 1][0]])
            y_pred = adj_cp_model.predict(x_seg)
            normalized_coef_list.append(
                100 * adj_cp_model.coef_ / data['key_indicator_filtered'].iloc[turning_points[i][0]])
            coef_list.append(adj_cp_model.coef_)
            y_pred_list.append(y_pred)

        # 3. While the chunk does not satisfy the length limit, and metric satisfied the merging_threshold
        #     1.merge the chunk with its neighbor
        #     2.recalculate the slope

        if self.merging_dynamic_constraint != float('inf'):
            print('Only merge dynamic <= distance: ', self.merging_dynamic_constraint)
        merging_round = 0
        if self.merging_threshold != -1:
            change = True
            while change and merging_round < 20:


                merging_round += 1
                counter = 0
                for i in range(len(turning_points) - 1):
                    if turning_points[i] != []:
                        counter += 1
                print('merging round: ', merging_round, 'current number of segments: ', counter)
                change = False

                # if we use the dynamic constraint, we would label the segment every time before merging
                if self.merging_dynamic_constraint !=float('inf'):
                    # calculate the slope
                    coef_list = []
                    normalized_coef_list = []
                    y_pred_list = []
                    indexs = []
                    turning_points_temp_flat = []
                    for i in range(len(turning_points) - 1):
                        if turning_points[i] == []:
                            continue
                        for j in range(i + 1, len(turning_points)):
                            if turning_points[j] != []:
                                next_index = j
                                break
                        x_seg = np.asarray([j for j in range(turning_points[i][0], turning_points[next_index][0])]).reshape(
                            -1, 1)
                        adj_cp_model = LinearRegression().fit(x_seg,
                                                              data['key_indicator_filtered'].iloc[
                                                              turning_points[i][0]:turning_points[next_index][0]])
                        y_pred = adj_cp_model.predict(x_seg)
                        normalized_coef_list.append(
                            100 * adj_cp_model.coef_ / data['key_indicator_filtered'].iloc[turning_points[i][0]])
                        coef_list.append(adj_cp_model.coef_)
                        y_pred_list.append(y_pred)
                        indexs.append(i)
                        turning_points_temp_flat.append(turning_points[i][0])

                    turning_points_temp_flat.append(turning_points[-1][0])
                    label, data_seg, label_seg_raw, index_seg = self.get_label(data=data, turning_points=turning_points_temp_flat,
                                                                           low=None, high=None, normalized_coef_list=normalized_coef_list, tic=tic,
                                                                           dynamic_num=self.dynamic_num,
                                                                           labeling_method=self.labeling_method)
                    # label the segments
                    label_seg=[None for _ in range(len(turning_points)-1)]
                    # print(indexs)
                    for i in range(len(indexs)):
                        label_seg[indexs[i]]=label_seg_raw[i]

                # record the distance
                distance_list=[]
                merge_prohibit_times=0
                # for every segment that does not reach self.max_length_expectation, calculate the the DTW distance between the segment and its neighbor
                for i in tqdm(range(len(turning_points) - 1)):
                    # find the first non-empty segment on right side
                    if turning_points[i] == []:
                        continue
                    have_next_index = False
                    for j in range(i + 1, len(turning_points)):
                        if turning_points[j] != []:
                            next_index = j
                            have_next_index = True
                            break
                    if have_next_index == False:
                        break
                    if turning_points[next_index][0] - turning_points[i][0] < self.max_length_expectation:
                        left_distance = float('inf')
                        right_distance = float('inf')
                        this_seg = data['key_indicator_filtered'].iloc[
                                   turning_points[i][0]:turning_points[next_index][0]].tolist()
                        if i > 0 and i < len(turning_points) - 1:
                            # the last turning point is the end of the data, and it should not be merged
                            # find the first non-empty segment on left side
                            left_index = None
                            for j in range(i - 1, -1, -1):
                                if turning_points[j] != []:
                                    left_index = j
                                    break
                            if left_index is not None:
                                left_neighbor = data['key_indicator_filtered'].iloc[
                                                turning_points[left_index][0]:turning_points[i][0]].tolist()
                                left_distance = self.calculate_distance(left_neighbor, this_seg, merging_round)
                        if i < len(turning_points) - 2:
                            # find the second non-empty segment on right side
                            next_index_2 = None
                            for j in range(next_index + 1, len(turning_points) - 1):
                                if turning_points[j] != []:
                                    next_index_2 = j
                                    break
                            if next_index_2 is not None:
                                right_neighbor = data['key_indicator_filtered'].iloc[
                                                 turning_points[next_index][0]:turning_points[next_index_2][0]].tolist()
                                right_distance = self.calculate_distance(this_seg, right_neighbor, merging_round)
                            else:
                                right_neighbor = data['key_indicator_filtered'].iloc[
                                                 turning_points[next_index][0]:].tolist()
                                right_distance = self.calculate_distance(this_seg, right_neighbor, merging_round)
                        # pick the min distance that is smaller than the threshold to merge
                        # may choose to merge with the shorter neighbor for balanced segment length

                        if left_distance!=float('inf'):
                            distance_list.append(left_distance)
                        if right_distance!=float('inf'):
                            distance_list.append(right_distance)

                        # if we activate the dynamic constraint
                        if self.merging_dynamic_constraint != float('inf'):
                            # check right
                            if next_index<len(label_seg) and right_distance!=float('inf') and self.merging_dynamic_constraint < abs(label_seg[i] - label_seg[next_index]):
                                if right_distance < self.merging_threshold:
                                    merge_prohibit_times+=1
                                    # print(f'prohibit merging right of {label_seg[i]} and {label_seg[next_index]}')
                                right_distance = float('inf')
                            # check left
                            if i > 0:
                                if left_distance!=float('inf') and self.merging_dynamic_constraint < abs(label_seg[i] - label_seg[left_index]):
                                    if left_distance < self.merging_threshold:
                                        merge_prohibit_times+=1
                                        # print(f'prohibit merging left of {label_seg[i]} and {label_seg[left_index]}')
                                    left_distance = float('inf')

                        if min(left_distance, right_distance) < self.merging_threshold:
                            # merge with the closer neighbor
                            if left_distance < right_distance:
                                turning_points[left_index] = turning_points[left_index] + turning_points[i]
                            else:
                                turning_points[next_index] = turning_points[i] + turning_points[next_index]
                            change = True
                            turning_points[i] = []
                print('All distance statistics:')
                print(pd.Series(distance_list).describe())
                print('Your merging_threshold is: ', self.merging_threshold)
                print(f'Merge prohibit times by merging_dynamic_constraint: {merge_prohibit_times}')



            # remove empty segments
            turning_points_new = []
            for i in range(len(turning_points)):
                if turning_points[i] != []:
                    turning_points_new.append(turning_points[i])
            turning_points = turning_points_new


            #log merging details to guide the tuning of parameters
            print(f'merging_round in total: {merging_round}, number of segments: {len(turning_points)}')
            #describe the distribution of distance by using pd.describe()
            print('You may want to tune the merging_threshold and merging_dynamic_constraint to get a better result.')


            # calculate the slope again
            coef_list = []
            normalized_coef_list = []
            y_pred_list = []
            for i in range(len(turning_points) - 1):
                x_seg = np.asarray([j for j in range(turning_points[i][0], turning_points[i + 1][0])]).reshape(-1, 1)
                adj_cp_model = LinearRegression().fit(x_seg,
                                                      data['key_indicator_filtered'].iloc[
                                                      turning_points[i][0]:turning_points[i + 1][0]])
                y_pred = adj_cp_model.predict(x_seg)
                normalized_coef_list.append(
                    100 * adj_cp_model.coef_ / data['key_indicator_filtered'].iloc[turning_points[i][0]])
                coef_list.append(adj_cp_model.coef_)
                y_pred_list.append(y_pred)

        # reshape turning_points to a 1d list
        turning_points = [i[0] for i in turning_points]


        return np.asarray(coef_list), np.asarray(turning_points), y_pred_list, normalized_coef_list

    def plot(self, tics, parameters, output_path, model_id):
        self.plot_path = os.path.join(output_path, model_id)
        # self.plot_path_filtered = os.path.join(os.path.dirname(os.path.realpath(data_path)), 'MDM_linear_filtered',
        #                                        model_id)
        # if not os.path.exists(self.plot_path):
        #     os.makedirs(self.plot_path)
        if self.method == 'slice_and_merge':
            try:
                low, high = parameters
            except:
                raise Exception(
                    "parameters shoud be [low,high] where the series would be split into 4 dynamics by low,high and 0 as threshold based on slope. A value of -0.5 and 0.5 stand for -0.5% and 0.5% change per step.")
            for tic in tics:
                paths = []
                paths.append(self.plot_to_file(self.data_dict[tic], tic, self.y_pred_dict[tic],
                                               self.turning_points_dict[tic], low, high,
                                               normalized_coef_list=self.norm_coef_list_dict[tic],
                                               plot_path=self.plot_path, plot_feather=self.key_indicator))
                self.plot_to_file(self.data_dict[tic], tic, self.y_pred_dict[tic],
                                  self.turning_points_dict[tic], low, high,
                                  normalized_coef_list=self.norm_coef_list_dict[tic],
                                  plot_path=self.plot_path, plot_feather='key_indicator_filtered',if_color=False,suffix='_denoised')
                self.plot_to_file(self.data_dict[tic], tic, self.y_pred_dict[tic],
                                  self.turning_points_dict[tic], low, high,
                                  normalized_coef_list=self.norm_coef_list_dict[tic],
                                  plot_path=self.plot_path, plot_feather=self.key_indicator,if_color=False,suffix='_original')
                # self.plot_to_file(self.data_dict[tic], tic, self.y_pred_dict[tic],
                #                   self.turning_points_dict[tic], low, high,
                #                   normalized_coef_list=self.norm_coef_list_dict[tic],
                #                   folder_name=self.plot_path_filtered, plot_feather='key_indicator_filtered')
            return paths
            try:
                print('plotting TSNE')
                self.TSNE_plot(self.tsne_results, self.all_label_seg, folder_name=self.plot_path)
            except:
                print('not able to plot TSNE')

    def plot_to_file(self, data, tic, y_pred_list, turning_points, low, high, normalized_coef_list,
                     plot_path=None, plot_feather=None,suffix='',if_color=True):
        data = data.reset_index(drop=True)
        # every sub-plot is contained segment of at most 100000 data points
        # 1. split the data into segments if the length is too long
        # For the case that the data of a long period may have significant value change, we split the data into segments (max of 4)

        segment_num = min(data.shape[0]/1000,4)
        segment_length = min(100000, data.shape[0]//segment_num)
        plot_segments = []
        counter = 0
        segments_buffer = [turning_points[0]]
        for index, j in enumerate(range(len(turning_points) - 1)):
            counter += turning_points[j + 1] - turning_points[j]
            segments_buffer.append(turning_points[j + 1])
            if counter > segment_length:
                plot_segments.append(segments_buffer)
                segments_buffer = [turning_points[j + 1]]
                counter = 0
        plot_segments.append(segments_buffer)
        sub_plot_num = len(plot_segments)

        fig, axs = plt.subplots(sub_plot_num, 1, figsize=(50, 15 * sub_plot_num), constrained_layout=True)
        if sub_plot_num == 1:
            axs = [axs]
        if if_color:
            colors = list(dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS).keys())
        else:
            colors = ['black' for _ in range(999)]

        counter = 0
        for index, ax in enumerate(axs):
            turning_points_seg = plot_segments[index]
            for i in range(len(turning_points_seg) - 1):
                x_seg = np.asarray([j for j in range(turning_points_seg[i], turning_points_seg[i + 1])]).reshape(-1, 1)
                coef = normalized_coef_list[i + counter]
                if self.labeling_method == 'slope' or self.labeling_method == 'quantile':
                    coef = coef[0]
                elif self.labeling_method == 'DTW':
                    coef = i + counter
                flag = self.dynamic_flag.get(coef)
                ax.plot(x_seg, data[plot_feather].iloc[turning_points_seg[i]:turning_points_seg[i + 1]],
                        color=colors[flag], label='Market Dynamics ' + str(flag), linewidth=3)
            counter += len(turning_points_seg) - 1
            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            font = font_manager.FontProperties(weight='bold',
                                               style='normal', size=48)
            # legend to every sub-plot
            ax.legend(by_label.values(), by_label.keys(), prop=font)
        # set the title
        plt.title(f"Dynamics_of_{tic}_linear_{self.labeling_method}_{plot_feather}", fontsize=20)
        fig_path = plot_path + '_' + tic+suffix + '.png'
        # print('plot to ' + fig_path)
        fig.savefig(fig_path)
        plt.close(fig)
        return os.path.abspath(fig_path).replace("\\", "/")

    def linear_regession_timewindow(self, data_ori, tic, adjcp_timewindow):
        # This is the version of linear regession that does not use a turning point to segment. Instead, it applys a fixed-length time winodw.
        # This can be helpful to process data that is extremely volatile, or you simply want a long-term dynamic. However you can achieve similar result by applying stronger filter.
        data = data_ori.iloc[:adjcp_timewindow * (data_ori['key_indicator_filtered'].size // adjcp_timewindow), :]
        adjcp_window_data = [
            data[['key_indicator_filtered']][i * adjcp_timewindow:(i + 1) * adjcp_timewindow].to_numpy().reshape(-1) for
            i in
            range(data['key_indicator_filtered'].size // adjcp_timewindow)]
        coef_list = []
        fig, ax = plt.subplots(2, 1, figsize=(20, 10), constrained_layout=True)
        ax[0].plot([i for i in range(data.shape[0])], data['key_indicator_filtered'])
        for i, data_seg in enumerate(adjcp_window_data):
            x_seg = np.asarray([i * adjcp_timewindow + j for j in range(adjcp_timewindow)]).reshape(-1, 1)
            adj_cp_model = LinearRegression().fit(x_seg, data_seg)
            y_pred = adj_cp_model.predict(x_seg)
            ax[1].plot([i * adjcp_timewindow + j for j in range(adjcp_timewindow)], y_pred)
            coef_list.append(adj_cp_model.coef_)
        return coef_list

    def interpolation(self, data):
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

    def TSNE_run(self, data_seg):
        interpolated_pct_return_data_seg = np.array(self.interpolation(data_seg))
        self.tsne = TSNE(n_components=2, perplexity=40, n_iter=300)
        self.tsne_results = self.tsne.fit_transform(interpolated_pct_return_data_seg)

    def TSNE_plot(self, data, label_list, title='', folder_name=None):
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
        fig.savefig(plot_path + 'TSNE' + title + '.png')
        plt.close(fig)
