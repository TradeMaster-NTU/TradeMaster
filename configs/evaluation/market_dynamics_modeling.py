task_name = "algorithmic_trading"
dataset_name = "BTC"


_base_ = [
    f"../_base_/market_dynamics_model/{task_name}/{dataset_name}/mdm.py",
]

market_dynamics_model = dict(
    type='Linear_Market_Dynamics_Model',
    data_path="data/high_frequency_trading/small_BTC/test.csv",
filter_strength=1,
slope_interval=[-0.01,0.01],
dynamic_number=3,
max_length_expectation=300,
OE_BTC=False,
PM='',
process_datafile_path='',
market_dynamic_labeling_visualization_paths='',
key_indicator='adjcp',
timestamp='date',
tic='tic',
labeling_method='slope',
min_length_limit=-1,
merging_metric='DTW_distance',
merging_threshold=-1,
merging_dynamic_constraint=-1
)