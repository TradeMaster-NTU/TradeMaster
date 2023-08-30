market_dynamics_model = dict(
    data_path='data/market_dynamics_modeling/second_level_BTC_LOB/data.feather',
    filter_strength=1,
    slope_interval=[1, -1],
    dynamic_number=5,
    max_length_expectation=3600,
    key_indicator='bid1_price',
    timestamp='timestamp',
    tic='BTCUSDT',
    labeling_method='quantile',
    min_length_limit=60,
    merging_metric='DTW_distance',
    merging_threshold=0.0003,
    merging_dynamic_constraint=1,
    OE_BTC=False,
    PM='',
    exp_name='BTC',
    process_datafile_path=
    '',
    market_dynamic_modeling_visualization_paths=[
    ],
    market_dynamic_modeling_analysis_paths=[
    ],
    type='Linear_Market_Dynamics_Model')
task_name = 'custom'
dataset_name = 'custom'
