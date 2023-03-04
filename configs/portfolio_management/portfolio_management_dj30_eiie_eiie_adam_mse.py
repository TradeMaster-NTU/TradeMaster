task_name = "portfolio_management"
dataset_name = "dj30"
net_name = "eiie"
agent_name = "eiie"
optimizer_name = "adam"
loss_name = "mse"
work_dir = f"work_dir/{task_name}_{dataset_name}_{net_name}_{agent_name}_{optimizer_name}_{loss_name}"

_base_ = [
    f"../_base_/datasets/{task_name}/{dataset_name}.py",
    f"../_base_/environments/{task_name}/env.py",
    f"../_base_/agents/{task_name}/{agent_name}.py",
    f"../_base_/trainers/{task_name}/eiie_trainer.py",
    f"../_base_/losses/{loss_name}.py",
    f"../_base_/optimizers/{optimizer_name}.py",
    f"../_base_/nets/{net_name}.py",
    f"../_base_/transition/transition.py"
]

data = dict(
    type='PortfolioManagementDataset',
    data_path='data/portfolio_management/dj30',
    train_path='data/portfolio_management/dj30/train.csv',
    valid_path='data/portfolio_management/dj30/valid.csv',
    test_path='data/portfolio_management/dj30/test.csv',
    test_dynamic_path='data/portfolio_management/dj30/DJI_label_by_DJIindex_3_24_-0.25_0.25.csv',
    tech_indicator_list=[
        'zopen', 'zhigh', 'zlow', 'zadjcp', 'zclose',
        'zd_5', 'zd_10', 'zd_15', 'zd_20', 'zd_25', 'zd_30'
    ],
    length_day=10,
    initial_amount=100000,
    transaction_cost_pct=0.001)

environment = dict(type='PortfolioManagementEIIEEnvironment')
transition = dict(
    type = "Transition"
)
agent = dict(
    type='PortfolioManagementEIIE',
    memory_capacity=1000,
    gamma=0.99,
    policy_update_frequency=500)

trainer = dict(
    type='PortfolioManagementEIIETrainer',
    epochs=2,
    work_dir=work_dir,
    if_remove=False )

loss = dict(type='MSELoss')

optimizer = dict(type='Adam', lr=0.001)

act = dict(
    type = "EIIEConv",
    input_dim = None,
    output_dim=1,
    time_steps=10,
    kernel_size=3,
    dims = [32]
)

cri = dict(
    type = "EIIECritic",
    input_dim = None,
    action_dim = None,
    output_dim=1,
    time_steps=None,
    num_layers = 1,
    hidden_size=32
)
