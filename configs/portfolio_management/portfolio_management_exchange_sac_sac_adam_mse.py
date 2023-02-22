task_name = "portfolio_management"
dataset_name = "exchange"
net_name = "sac"
agent_name = "sac"
optimizer_name = "adam"
loss_name = "mse"
work_dir = f"work_dir/{task_name}_{dataset_name}_{net_name}_{agent_name}_{optimizer_name}_{loss_name}"

_base_ = [
    f"../_base_/datasets/{task_name}/{dataset_name}.py",
    f"../_base_/environments/{task_name}/env.py",
    f"../_base_/trainers/{task_name}/trainer.py",
    f"../_base_/losses/{loss_name}.py",
    f"../_base_/optimizers/{optimizer_name}.py",
]

data = dict(
    type = "PortfolioManagementDataset",
    data_path = "data/portfolio_management/exchange",
    train_path = "data/portfolio_management/exchange/train.csv",
    valid_path = "data/portfolio_management/exchange/valid.csv",
    test_path = "data/portfolio_management/exchange/test.csv",
    test_dynamic_path='data/portfolio_management/exchange/test_labeled_3_24_-0.05_0.05.csv',
    tech_indicator_list = [
        "zopen",
        "zhigh",
        "zlow",
        "zadjcp",
        "zclose",
        "zd_5",
        "zd_10",
        "zd_15",
        "zd_20",
        "zd_25",
        "zd_30"
    ],
    initial_amount = 100000,
    transaction_cost_pct = 0.001
)

trainer = dict(
    agent_name = agent_name,
    work_dir = work_dir
)