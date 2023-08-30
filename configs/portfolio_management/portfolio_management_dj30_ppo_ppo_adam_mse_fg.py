task_name = "portfolio_management"
dataset_name = "dj30"
net_name = "ppo"
agent_name = "ppo"
optimizer_name = "adam"
loss_name = "mse"
work_dir = f"work_dir/{task_name}_{dataset_name}_{net_name}_{agent_name}_{optimizer_name}_{loss_name}_fg"

_base_ = [
    f"../_base_/datasets/{task_name}/{dataset_name}.py",
    f"../_base_/environments/{task_name}/env.py",
    f"../_base_/trainers/{task_name}/trainer.py",
    f"../_base_/losses/{loss_name}.py",
    f"../_base_/optimizers/{optimizer_name}.py",
]

data = dict(
    type = "PortfolioManagementDataset",
    data_path = "data/portfolio_management/dj30",
    train_path = "data/portfolio_management/dj30/new_train.csv",
    valid_path = "data/portfolio_management/dj30/new_valid.csv",
    test_path = "data/portfolio_management/dj30/new_test.csv",
    test_dynamic_path='data/portfolio_management/dj30/test_with_label.csv',
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
        "zd_30",
        "autoFE_f_0", 
        "autoFE_f_1", 
        "autoFE_f_2",
        "autoFE_f_3", 
        "autoFE_f_4", 
        "autoFE_f_5", 
        "autoFE_f_6", 
        "autoFE_f_7",
        "autoFE_f_8", 
        "autoFE_f_9"
    ],
    initial_amount = 100000,
    transaction_cost_pct = 0.001
)

trainer = dict(
    agent_name = agent_name,
    work_dir = work_dir,
    epochs=2,
)