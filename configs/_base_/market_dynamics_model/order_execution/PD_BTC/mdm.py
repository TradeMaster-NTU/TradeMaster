market_dynamics_model = dict(
    data_path="data/order_execution/PD_BTC/test.csv",
filter_strength=1,
labeling_parameters=[-0.15,0.15],
dynamic_number=3,
length_limit=24,
OE_BTC=False,
PM='',
process_datafile_path='',
market_dynamic_labeling_visualization_paths='',
key_indicator='adjcp',
timestamp='date',
tic='tic',
mode='slope',
hard_length_limit=-1,
merging_metric='DTW_distance',
merging_threshold=-1
)