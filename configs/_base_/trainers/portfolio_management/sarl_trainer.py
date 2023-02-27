trainer = dict(
    type="PortfolioManagementSARLTrainer",
    agent_name="ddpg",
    if_remove=False ,
    configs = {
        "framework" : 'tf2'
    },
    work_dir="work_dir",
)
