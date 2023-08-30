market_dynamics_model = dict(
    data_path='data/high_frequency_trading/small_BTC/test.csv',
    filter_strength=1,
    slope_interval=[1, -1],
    dynamic_number=3,
    max_length_expectation=480,
    key_indicator='bid1_price',
    timestamp='timestamp',
    tic='BTC',
    labeling_method='quantile',
    min_length_limit=48,
    merging_metric='DTW_distance',
    merging_threshold=0.0003,
    merging_dynamic_constraint=1,
    OE_BTC=False,
    PM='',
    exp_name='Market_Dynamics_Model',
    process_datafile_path=
    '/home/hcxia/trademaster_new/TradeMaster/data/high_frequency_trading/small_BTC/Market_Dynamics_Model/BTC/test_labeled_slice_and_merge_model_3dynamics_minlength48_quantile_labeling.csv',
    market_dynamic_modeling_visualization_paths=[
        '/home/hcxia/trademaster_new/TradeMaster/data/high_frequency_trading/small_BTC/Market_Dynamics_Model/BTC/slice_and_merge_model_3dynamics_minlength48_quantile_labeling_HFT_small_BTC.png'
    ],
    market_dynamic_modeling_analysis_paths=[
        'data/high_frequency_trading/small_BTC/Market_Dynamics_Model/BTC/default/metrics_of_each_dynamics.csv'
    ],
    type='Linear_Market_Dynamics_Model')
task_name = 'custom'
dataset_name = 'custom'
