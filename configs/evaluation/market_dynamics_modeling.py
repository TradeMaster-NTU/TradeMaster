task_name = "algorithmic_trading"
dataset_name = "BTC"


_base_ = [
    f"../_base_/market_dynamics_model/{task_name}/{dataset_name}/mdm.py",
]

market_dynamics_model = dict(
    type='Linear_Market_Dynamics_Model',
    data_path="data/algorithmic_trading/BTC/test.csv",
fitting_parameters=['2/7','2/14','4'],
labeling_parameters=[-0.15,0.15],
regime_number=3,
length_limit=24,
OE_BTC=False,
PM=''
)