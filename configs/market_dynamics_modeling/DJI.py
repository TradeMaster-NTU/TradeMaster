market_dynamics_model = dict(
    data_path='data/portfolio_management/dj30/DJI.csv',
    filter_strength=1,
    slope_interval=[1, -1],
    dynamic_number=3,
    max_length_expectation=120,
    key_indicator='adjcp',
    timestamp='date',
    tic='DJI',
    labeling_method='quantile',
    min_length_limit=12,
    merging_metric='DTW_distance',
    merging_threshold=0.003,
    merging_dynamic_constraint=1,
    OE_BTC=False,
    PM='',
    exp_name='Market_Dynamics_Model',
    process_datafile_path=
    '/home/hcxia/trademaster_new/TradeMaster/data/portfolio_management/dj30/Market_Dynamics_Model/DJI/DJI_labeled_slice_and_merge_model_3dynamics_minlength12_quantile_labeling.csv',
    market_dynamic_modeling_visualization_paths=[
        '/home/hcxia/trademaster_new/TradeMaster/data/portfolio_management/dj30/Market_Dynamics_Model/DJI/slice_and_merge_model_3dynamics_minlength12_quantile_labeling_DJI.png'
    ],
    market_dynamic_modeling_analysis_paths=[
        'data/portfolio_management/dj30/Market_Dynamics_Model/DJI/DJI/metrics_of_each_dynamics.csv'
    ],
    type='Linear_Market_Dynamics_Model')
task_name = 'custom'
dataset_name = 'custom'
