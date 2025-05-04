data = dict(
    type='PortfolioManagementDataset',
    data_path='data/portfolio_management/dj30',
    train_path='data/portfolio_management/dj30/train.csv',
    valid_path='data/portfolio_management/dj30/valid.csv',
    test_path='data/portfolio_management/dj30/test.csv',
    tech_indicator_list=[
        'high', 'low', 'open', 'close', 'adjcp', 'zopen', 'zhigh', 'zlow',
        'zadjcp', 'zclose', 'zd_5', 'zd_10', 'zd_15', 'zd_20', 'zd_25', 'zd_30'
    ],
    length_day=5,
    initial_amount=10000,
    transaction_cost_pct=0.001,
    test_dynamic_path='data/portfolio_management/dj30/test_with_label.csv',
    test_dynamic='-1')
environment = dict(type='PortfolioManagementSARLEnvironment')
trainer = dict(
    type='PortfolioManagementSARLTrainer',
    agent_name='ddpg',
    if_remove=False,
    configs=dict(framework='tf2', num_workers=0),
    work_dir='work_dir/portfolio_management_dj30_sarl_sarl_adam_mse',
    epochs=2)
loss = dict(type='MSELoss')
optimizer = dict(type='Adam', lr=0.001)
task_name = 'portfolio_management'
dataset_name = 'dj30'
net_name = 'sarl'
agent_name = 'sarl'
optimizer_name = 'adam'
loss_name = 'mse'
work_dir = 'work_dir/portfolio_management_dj30_sarl_sarl_adam_mse'
