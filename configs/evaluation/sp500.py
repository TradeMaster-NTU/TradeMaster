market_dynamics_model = dict(
    data_path='/home/hcxia/Style-timegan/timegan-pytorch/data/sp500/features',
    filter_strength=1,
    slope_interval=[-0.01, 0.01],
    dynamic_number=5,
    max_length_expectation=140,
    OE_BTC=False,
    PM='',
    process_datafile_path=
    '/home/hcxia/Style-timegan/timegan-pytorch/data/sp500/features/MTB/MTB_labeled_slice_and_merge_model_5dynamics_minlength24_quantile_labeling.csv',
    market_dynamic_labeling_visualization_paths=[
        '/home/hcxia/Style-timegan/timegan-pytorch/data/sp500/features/MTB/slice_and_merge_model_5dynamics_minlength24_quantile_labeling_MTB.png'
    ],
    key_indicator='Adj Close',
    timestamp='Date',
    tic='',
    labeling_method='quantile',
    min_length_limit=24,
    merging_metric='DTW_distance',
    merging_threshold=0.03,
    merging_dynamic_constraint=1,
    type='Linear_Market_Dynamics_Model')
task_name = 'custom'
dataset_name = 'custom'
