task_name = "market_dynamics_modeling"
dataset_name = "custom"
optimizer_name = "adam"
loss_name = "mse"
net_name = "dqn"
agent_name = "Null"
work_dir = f"work_dir/{task_name}_{dataset_name}_{net_name}_{agent_name}_{optimizer_name}_{loss_name}"

_base_ = [
    f"../_base_/datasets/{task_name}/{dataset_name}.py",
    f"../_base_/environments/{task_name}/env.py",
    f"../_base_/agents/{task_name}/{agent_name}.py",
    f"../_base_/trainers/{task_name}/trainer.py",
    f"../_base_/losses/{loss_name}.py",
    f"../_base_/optimizers/{optimizer_name}.py",
    f"../_base_/nets/{net_name}.py",
    f"../_base_/transition/transition.py"
]

batch_size = 64
data = dict(
    type='Market_Dynamics_Modeling_Dataset',
    data_path='',
    train_path='',
    valid_path='',
    test_path='',
    test_dynamic_path=
    '',
    tech_indicator_list=[
    ])
environment = dict(type='Market_Dynamics_ModelingENV')
transition = dict(
    type = "Transition"
)
agent = dict(
    type='Null',
)
trainer = dict(
    type='Null')
loss = dict(type='MSELoss')
optimizer = dict(type='Adam', lr=0.001)
act = dict(
    type='QNet', state_dim=82, action_dim=3, dims=(64, 32), explore_rate=0.25)
cri = None