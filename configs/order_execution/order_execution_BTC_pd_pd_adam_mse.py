task_name = "order_execution"
dataset_name = "BTC"
net_name = "pd"
agent_name = "pd"
optimizer_name = "adam"
loss_name = "mse"
work_dir = f"work_dir/{task_name}_{dataset_name}_{net_name}_{agent_name}_{optimizer_name}_{loss_name}"

_base_ = [
    f"../_base_/datasets/{task_name}/{dataset_name}.py",
    f"../_base_/environments/{task_name}/env.py",
    f"../_base_/agents/{task_name}/{agent_name}.py",
    f"../_base_/trainers/{task_name}/pd_trainer.py",
    f"../_base_/losses/{loss_name}.py",
    f"../_base_/optimizers/{optimizer_name}.py",
    f"../_base_/nets/{net_name}.py",
    f"../_base_/transition/transition.py"
]

data = dict(
    type="OrderExecutionDataset",
    data_path="data/order_execution/PD_BTC",
    train_path="data/order_execution/PD_BTC/train.csv",
    valid_path="data/order_execution/PD_BTC/valid.csv",
    test_path="data/order_execution/PD_BTC/test.csv",
    test_style_path="data/order_execution/PD_BTC/test_labeled_3_24_-0.15_0.15.csv",
    tech_indicator_list=[
        "high",
        "low",
        "open",
        "close",
        "adjcp",
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
    ],
    length_keeping=30,
    state_length=10,
    target_order=1,
    initial_amount=100000,
)

environment = dict(
    type="OrderExecutionPDEnvironment",
)

transition = dict(
    type = "TransitionPD"
)

agent = dict(
    type="OrderExecutionPD",
    memory_capacity=100,
    memory_update_freq = 10,
    gamma=0.9,
    climp=0.2,
    sample_effiency=0.5,
)

trainer = dict(
    type="OrderExecutionPDTrainer",
    epochs=10,
    work_dir=work_dir,
    if_remove=True,
)

loss = dict(type='MSELoss')

optimizer = dict(type='Adam', lr=0.001)

act = dict(
    type="PDNet",
    input_feature = None,
    hidden_size = 32,
    private_feature = None
)

cri = dict(
    type="PDNet",
    input_feature = None,
    hidden_size = 32,
    private_feature = None
)
