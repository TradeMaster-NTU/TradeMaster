import pandas as pd
import numpy as np
import os
import re
import argparse
import matplotlib.pyplot as plt
class MarketDynamicsModelingAnalysis(object):
    def __init__(self, data_path, key_indicator):
        self.data_path = data_path
        self.key_indicator = key_indicator
        # get extension of data_path, without the dot
        self.file_extension = os.path.splitext(self.data_path)[1][1:]
    def sort_list(self,lst: list):
        convert = lambda text: int(text) if text.isdigit() else text
        alphanum_key = lambda key: [convert(c) for c in re.split("([0-9]+)", key)]
        lst.sort(key=alphanum_key)


    def calculate_mdd_k(self,df):
        price_list = [df.iloc[0][self.key_indicator]]
        for i in range(1, len(df)):
            price_list.append(df.iloc[i][self.key_indicator])
        mdd = 0
        peak = price_list[0]
        previous_mdd_start_step = 0
        previous_mdd_end_step = 0
        current_dd_start_step = 0
        current_dd_end_step = 0
        for i, value in zip(range(len(price_list)), price_list):
            if value > peak:
                peak = value
                current_dd_start_step = i
            dd = (peak - value) / peak
            current_dd_end_step = i
            if dd > mdd:
                mdd = dd
                previous_mdd_start_step = current_dd_start_step
                previous_mdd_end_step = current_dd_end_step
        return (
            mdd / (previous_mdd_end_step - previous_mdd_start_step+1e-10),
            (previous_mdd_end_step - previous_mdd_start_step),
            (previous_mdd_end_step - previous_mdd_start_step) / len(df),
            mdd/len(df)
        )


    def calculate_mpp_k(self,df):
        price_list = [df.iloc[0][self.key_indicator]]
        for i in range(1, len(df)):
            price_list.append(df.iloc[i][self.key_indicator])
        mpp = 0
        nadir = price_list[0]
        previous_mpp_start_step = 0
        previous_mpp_end_step = 0
        current_pp_start_step = 0
        current_pp_end_step = 0
        for i, value in zip(range(len(price_list)), price_list):
            if value < nadir:
                nadir = value
                current_pp_start_step = i
            pp = (value - nadir) / nadir
            current_pp_end_step = i
            if pp > mpp:
                mpp = pp
                previous_mpp_start_step = current_pp_start_step
                previous_mpp_end_step = current_pp_end_step

        return (
            mpp / (previous_mpp_end_step - previous_mpp_start_step+1e-10),
            (previous_mpp_end_step - previous_mpp_start_step),
            (previous_mpp_end_step - previous_mpp_start_step) / len(df),
            mpp/len(df)
        )


    # def calculate_average_k(self,df):
    #     price_list = [df.iloc[0][self.key_indicator]]
    #     for i in range(1, len(df)):
    #         price_list.append(df.iloc[i][self.key_indicator])
    #
    #     return (price_list[-1] - price_list[0]) / (len(price_list) * price_list[0])

    def calculate_average_k(self,df):
        price_list = [df.iloc[0][self.key_indicator]]
        for i in range(1, len(df)):
            price_list.append(df.iloc[i][self.key_indicator])
        # slope=(price_list[-1] - price_list[0]) / (len(price_list) * price_list[0])
        # calculate average slope by linear regression
        x = np.arange(len(price_list))
        y = np.array(price_list)
        slope = np.polyfit(x, y, 1)[0]
        slope=slope/price_list[0]
        return slope



    def get_intervals(self,data):
        # if no index column, add index column
        index=data['index']
        #cast index to list
        index=index.tolist()
        last_value=index[0]-1
        last_index=0
        intervals=[]
        for i in range(data.shape[0]):
            if last_value!=index[i]-1:
                intervals.append([last_index,i])
                last_value=index[i]
                last_index=i
            last_value=index[i]
        intervals.append([last_index, i])
        return intervals

    def save_data_by_dynamics(self,data,data_folder):

        # segment data into dynamics with 'label' column
        # get unique label
        dynamics = data['label'].unique()

        # get folder of data_path
        # create folder for each dynamics under data_folder
        for i in range(len(dynamics)):
            if not os.path.exists(data_folder + '/label_' + str(i)):
                os.makedirs(data_folder + '/label_' + str(i))

        # for each dynamics, segment data into slices that have consecutive same label
        for i in range(len(dynamics)):
            dynamic_data = data.loc[data['label'] == i, :]
            intervals=self.get_intervals(dynamic_data)
            for index,interval in enumerate(intervals):
                dynamic_data_seg = dynamic_data.iloc[interval[0]:interval[1], :]
                # save segmented data to file
                if self.file_extension == 'feather':
                    dynamic_data_seg.reset_index().to_feather(data_folder + '/label_' + str(i) + '/label_' + str(i) + '_' + str(index) + '.' + self.file_extension)
                elif self.file_extension == 'csv':
                    dynamic_data_seg.to_csv(data_folder + '/label_' + str(i)  + '/label_' + str(i) + '_' + str(index) +'.'+self.file_extension)
        return data_folder,len(dynamics)



    def calculate_metrics(self,dynamics_num, data_folder):
        dynamics_num=int(dynamics_num)
        average_k_list_list = []
        average_length_list_list = []
        mpp_k_list_list = []
        mdd_k_list_list = []
        mpp_length_list_list = []
        mdd_length_list_list = []
        mpp_percentile_list_list = []
        mdd_percentile_list_list = []
        mpp_sum_percentile_list_list = []
        mdd_sum_percentile_list_list = []
        for label in range(dynamics_num):
            average_k_list = []
            average_length_list = []
            mpp_k_list = []
            mdd_k_list = []
            mpp_length_list = []
            mdd_length_list = []
            mpp_percentile_list = []
            mdd_percentile_list = []
            mpp_sum_percentile_list=[]
            mdd_sum_percentile_list=[]
            test_df_path = f"{data_folder}/label_{label}"
            df_list = os.listdir(test_df_path)
            self.sort_list(df_list)
            for df in df_list:
                if self.file_extension == 'feather':
                    df_result = pd.read_feather(os.path.join(test_df_path, df))
                elif self.file_extension == 'csv':
                    df_result = pd.read_csv(os.path.join(test_df_path, df))
                df_result.drop(columns=["index"], inplace=True)
                average_k_list.append(self.calculate_average_k(df_result))
                average_length_list.append(len(df_result))
                mpp_k_list.append(self.calculate_mpp_k(df_result)[0])
                mdd_k_list.append(self.calculate_mdd_k(df_result)[0])
                mpp_length_list.append(self.calculate_mpp_k(df_result)[1])
                mdd_length_list.append(self.calculate_mdd_k(df_result)[1])
                mpp_percentile_list.append(self.calculate_mpp_k(df_result)[2])
                mdd_percentile_list.append(self.calculate_mdd_k(df_result)[2])
                mpp_sum_percentile_list.append(self.calculate_mpp_k(df_result)[3])
                mdd_sum_percentile_list.append(self.calculate_mdd_k(df_result)[3])
            average_k_list_list.append(average_k_list)
            average_length_list_list.append(average_length_list)
            mpp_k_list_list.append(mpp_k_list)
            mdd_k_list_list.append(mdd_k_list)
            mpp_length_list_list.append(mpp_length_list)
            mdd_length_list_list.append(mdd_length_list)
            mpp_percentile_list_list.append(mpp_percentile_list)
            mdd_percentile_list_list.append(mdd_percentile_list)
            mpp_sum_percentile_list_list.append(mpp_sum_percentile_list)
            mdd_sum_percentile_list_list.append(mdd_sum_percentile_list)
        # for i in range(dynamics_num):
        #     print("For label {}".format(i))
        #     print("average_k_list mean", np.mean(average_k_list_list[i]))
        #     print("average_k_list std", np.std(average_k_list_list[i]))
        #     print('average_length_list mean', np.mean(average_length_list_list[i]))
        #     print('average_length_list std', np.std(average_length_list_list[i]))
        #     print("mpp_k_list mean", np.mean(mpp_k_list_list[i]))
        #     print("mpp_k_list std", np.std(mpp_k_list_list[i]))
        #     print("mdd_k_list mean", np.mean(mdd_k_list_list[i]))
        #     print("mdd_k_list std", np.std(mdd_k_list_list[i]))
        #
        #     print("mpp_length_list mean", np.mean(mpp_length_list_list[i]))
        #     print("mpp_length_list std", np.std(mpp_length_list_list[i]))
        #     print("mdd_length_list mean", np.mean(mdd_length_list_list[i]))
        #     print("mdd_length_list std", np.std(mdd_length_list_list[i]))
        #
        #     print("mpp_quantile_list mean", np.mean(mpp_percentile_list_list[i]))
        #     print("mpp_quantile_list std", np.std(mpp_percentile_list_list[i]))
        #     print("mdd_quantile_list mean", np.mean(mdd_percentile_list_list[i]))
        #     print("mdd_quantile_list std", np.std(mdd_percentile_list_list[i]))
        #     print("=====================================")

        # plot mean of average_k_list, average_length_list, mpp_k_list, mdd_k_list, mpp_length_list, mdd_length_list of each label in a boxplot, each in a subplot

        fig, axs = plt.subplots(3, 2, figsize=(10, 10))
        # fig.suptitle('Metrics of each dynamics', fontsize=16)
        axs[0, 0].boxplot(average_k_list_list,showfliers=False)
        axs[0, 0].set_ylabel('Average slope')
        axs[0, 0].set_xlabel('label')
        axs[0, 0].set_xticks([i for i in range(dynamics_num+1)])
        axs[0, 0].set_xticklabels(['']+[i for i in range(dynamics_num)])
        axs[0, 0].set_title('Average slope of each dynamics')
        axs[0, 1].boxplot(average_length_list_list,showfliers=False)
        axs[0, 1].set_ylabel('Average length')
        axs[0, 1].set_xlabel('label')
        axs[0, 1].set_xticks([i for i in range(dynamics_num+1)])
        axs[0, 1].set_xticklabels(['']+[i for i in range(dynamics_num)])
        axs[0, 1].set_title('Average length of each dynamics')


        # axs[1, 0].boxplot(mpp_k_list_list,showfliers=False)
        # axs[1, 0].set_ylabel('Average max uptrend slope')
        # axs[1, 0].set_xlabel('label')
        # axs[1, 0].set_xticks([i for i in range(dynamics_num+1)])
        # axs[1, 0].set_xticklabels(['']+[i for i in range(dynamics_num)])
        # axs[1, 0].set_title('Average max uptrend slope of each dynamics')
        # axs[1, 1].boxplot(mdd_k_list_list,showfliers=False)
        # axs[1, 1].set_ylabel('Average max downtrend slope')
        # axs[1, 1].set_xlabel('label')
        # axs[1, 1].set_xticks([i for i in range(dynamics_num+1)])
        # axs[1, 1].set_xticklabels(['']+[i for i in range(dynamics_num)])
        # axs[1, 1].set_title('Average max downtrend slope of each dynamics')
        axs[1, 0].boxplot(mpp_length_list_list,showfliers=False)
        axs[1, 0].set_ylabel('Average max uptrend length')
        axs[1, 0].set_xlabel('label')
        axs[1, 0].set_xticks([i for i in range(dynamics_num+1)])
        axs[1, 0].set_xticklabels(['']+[i for i in range(dynamics_num)])
        axs[1, 0].set_title('Average max uptrend length of each dynamics')
        axs[1, 1].boxplot(mdd_length_list_list,showfliers=False)
        axs[1, 1].set_ylabel('Average max downtrend length')
        axs[1, 1].set_xlabel('label')
        axs[1, 1].set_xticks([i for i in range(dynamics_num+1)])
        axs[1, 1].set_xticklabels(['']+[i for i in range(dynamics_num)])
        axs[1, 1].set_title('Average max downtrend length of each dynamics')




        axs[2, 0].boxplot(mpp_sum_percentile_list_list,showfliers=False)
        axs[2, 0].set_ylabel('Average max uptrend')
        axs[2, 0].set_xlabel('label')
        axs[2, 0].set_xticks([i for i in range(dynamics_num+1)])
        axs[2, 0].set_xticklabels(['']+[i for i in range(dynamics_num)])
        axs[2, 0].set_title('Average max uptrend of each dynamics')
        axs[2, 1].boxplot(mdd_sum_percentile_list_list,showfliers=False)
        axs[2, 1].set_ylabel('Average max downtrend')
        axs[2, 1].set_xlabel('label')
        axs[2, 1].set_xticks([i for i in range(dynamics_num+1)])
        axs[2, 1].set_xticklabels(['']+[i for i in range(dynamics_num)])
        axs[2, 1].set_title('Average max downtrend of each dynamics')
        plt.tight_layout()
        # save the figure
        path=os.path.join(data_folder,'metrics_of_each_dynamics.png')
        fig.savefig(path)
        # print("metrics_of_each_dynamics.png saved at",path)
        # stor the numercial data to csv
        path = os.path.join(data_folder, 'metrics_of_each_dynamics.csv')
        # store the average and std of average_k_list_list, average_length_list_list, mpp_k_list_list, mdd_k_list_list, mpp_length_list_list, mdd_length_list_list to csv
        df = pd.DataFrame({'average_k_list_mean': [np.mean(average_k_list_list[i]) for i in range(dynamics_num)],
                           'average_k_list_std': [np.std(average_k_list_list[i]) for i in range(dynamics_num)],
                           'average_length_list_mean': [np.mean(average_length_list_list[i]) for i in
                                                        range(dynamics_num)],
                           'average_length_list_std': [np.std(average_length_list_list[i]) for i in
                                                       range(dynamics_num)],
                           'mpp_k_list_mean': [np.mean(mpp_k_list_list[i]) for i in range(dynamics_num)],
                           'mpp_k_list_std': [np.std(mpp_k_list_list[i]) for i in range(dynamics_num)],
                           'mdd_k_list_mean': [np.mean(mdd_k_list_list[i]) for i in range(dynamics_num)],
                           'mdd_k_list_std': [np.std(mdd_k_list_list[i]) for i in range(dynamics_num)],
                           'mpp_length_list_mean': [np.mean(mpp_length_list_list[i]) for i in range(dynamics_num)],
                           'mpp_length_list_std': [np.std(mpp_length_list_list[i]) for i in range(dynamics_num)],
                           'mdd_length_list_mean': [np.mean(mdd_length_list_list[i]) for i in range(dynamics_num)],
                           'mdd_length_list_std': [np.std(mdd_length_list_list[i]) for i in range(dynamics_num)]})
        df.to_csv(path)
        return path

    def run_analysis(self,data_path):
        # if extention is .feather
        if self.file_extension == 'feather':
            data = pd.read_feather(data_path).reset_index()
        # if extention is .csv
        elif self.file_extension == 'csv':
            data = pd.read_csv(data_path).reset_index()
        # get unique tic
        if 'tic' not in data.columns:
            data['tic']='default'
        tics = data['tic'].unique()
        market_dynamic_modeling_analysis_paths= []
        for tic in tics:
            # get data of each tic
            data_tic = data[data['tic'] == tic]
            data_folder = os.path.dirname(data_path) + '/' + tic
            data_folder,dynamics_num=self.save_data_by_dynamics(data_tic,data_folder)
            market_dynamic_modeling_analysis_path=self.calculate_metrics(dynamics_num,data_folder)
            market_dynamic_modeling_analysis_paths.append(market_dynamic_modeling_analysis_path)
        return market_dynamic_modeling_analysis_paths

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default='')
    parser.add_argument("--key_indicator", type=str, default='bid1_price')
    args = parser.parse_args()
    MarketDynamicsModelingAnalysis(args.data_path,args.key_indicator).run_analysis(args.data_path)

    

