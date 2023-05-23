market_dynamics_model = dict(
    data_path="data/order_execution/BTC/test.csv",
filter_strength=1,
slope_interval=[-0.01,0.01],
dynamic_number=3,
max_length_expectation=32,
OE_BTC=True,
PM='',
process_datafile_path='',
market_dynamic_labeling_visualization_paths='',
key_indicator='adjcp',
timestamp='date',
tic='tic',
labeling_method='slope',
min_length_limit=-1,
merging_metric='DTW_distance',
merging_threshold=-1
)