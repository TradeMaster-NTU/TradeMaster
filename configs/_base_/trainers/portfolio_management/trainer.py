trainer = dict(
    type="PortfolioManagementTrainer",
    agent_name="ppo",
    if_remove=False ,
    configs = {
        "framework" : 'tf2'
    },
    work_dir="work_dir",
)
