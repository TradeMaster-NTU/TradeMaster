market_dynamics_model = dict(
    data_path='data/algorithmic_trading/BTC/test.csv',
    filter_strength=1,
    slope_interval=[1, -1],
    dynamic_number=3,
    max_length_expectation=120,
    key_indicator='adjcp',
    timestamp='date',
    tic='BTC',
    labeling_method='quantile',
    min_length_limit=12,
    merging_metric='DTW_distance',
    merging_threshold=0.03,
    merging_dynamic_constraint=1,
    OE_BTC=False,
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
