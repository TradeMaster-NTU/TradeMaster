data = dict(
    type='PortfolioManagementDataset',
    data_path='data/portfolio_management/dj30',
    train_path='data/portfolio_management/dj30/train.csv',
    valid_path='data/portfolio_management/dj30/valid.csv',
    test_path='data/portfolio_management/dj30/test.csv',
    tech_indicator_list=[
        'zopen', 'zhigh', 'zlow', 'zadjcp', 'zclose', 'zd_5', 'zd_10', 'zd_15',
        'zd_20', 'zd_25', 'zd_30'
    ],
    length_day=10,
    initial_amount=100000,
    transaction_cost_pct=0.001,
    test_dynamic_path='data/portfolio_management/dj30/test_with_label.csv',
    test_dynamic='-1')
environment = dict(type='PortfolioManagementEIIEEnvironment')
agent = dict(
    type='PortfolioManagementEIIE',
    memory_capacity=1000,
    gamma=0.99,
    policy_update_frequency=500)
trainer = dict(
    type='PortfolioManagementEIIETrainer',
    epochs=2,
    work_dir='work_dir/portfolio_management_dj30_eiie_eiie_adam_mse',
    if_remove=False)
loss = dict(type='MSELoss')
optimizer = dict(type='Adam', lr=0.001)
act = dict(
    type='EIIEConv',
    input_dim=11,
    output_dim=1,
    time_steps=10,
    kernel_size=3,
    dims=[32])
cri = dict(
    type='EIIECritic',
    input_dim=11,
    action_dim=29,
    output_dim=1,
    time_steps=10,
    num_layers=1,
    hidden_size=32)
transition = dict(type='Transition')
task_name = 'portfolio_management'
dataset_name = 'dj30'
net_name = 'eiie'
agent_name = 'eiie'
optimizer_name = 'adam'
loss_name = 'mse'
work_dir = 'work_dir/portfolio_management_dj30_eiie_eiie_adam_mse'
