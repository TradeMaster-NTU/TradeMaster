trainer = dict(
    type="PortfolioManagementTrainer",
    agent_name="ppo",
    if_remove=False ,
    configs = {
        "framework" : 'tf2',
        "num_workers": 0,
    },
    work_dir="work_dir",
)
