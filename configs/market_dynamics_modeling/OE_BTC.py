market_dynamics_model = dict(
    data_path='data/order_execution/BTC/test.csv',
    filter_strength=1,
    slope_interval=[1, -1],
    dynamic_number=3,
    max_length_expectation=120,
    key_indicator='midpoint',
    timestamp='system_time',
    tic='OE_BTC',
    labeling_method='quantile',
    min_length_limit=12,
    merging_metric='DTW_distance',
    merging_threshold=0.0003,
    merging_dynamic_constraint=1,
    OE_BTC=True,
    PM='',
    exp_name='Market_Dynamics_Model',
    process_datafile_path=
    '',
    market_dynamic_modeling_visualization_paths=[
    ],
    market_dynamic_modeling_analysis_paths=[
    ],
    type='Linear_Market_Dynamics_Model')
task_name = 'custom'
dataset_name = 'custom'
