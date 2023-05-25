task_name = "custom"
dataset_name = "custom"


_base_ = [
    f"../_base_/market_dynamics_model/{task_name}/{dataset_name}/mdm.py",
]

market_dynamics_model = dict(
    type='Linear_Market_Dynamics_Model',
    data_path="data/market_dynamics_modeling/binance-futures_book_snapshot_5_2020-09-01_BTCUSDT.csv",
filter_strength=1,
slope_interval=[-0.01,0.01],
dynamic_number=5,
max_length_expectation=3600,
OE_BTC=False,
PM='',
process_datafile_path='',
market_dynamic_labeling_visualization_paths='',
key_indicator='bid1_price',
timestamp='timestamp',
tic='BTCUSDT',
labeling_method='quantile',
min_length_limit=60,
merging_metric='DTW_distance',
merging_threshold=0.0003,
merging_dynamic_constraint=1
)