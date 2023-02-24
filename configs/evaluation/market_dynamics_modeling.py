task_name = "algorithmic_trading"
dataset_name = "BTC"


_base_ = [
    f"../_base_/market_dynamics_model/{task_name}/{dataset_name}/mdm.py",
]

market_dynamics_model = dict(
    type='Linear_Market_Dynamics_Model',
    data_path="data/high_frequency_trading/small_BTC/test.csv",
fitting_parameters=['2/7','2/14','4'],
labeling_parameters=[-0.01,0.01],
regime_number=3,
length_limit=300,
OE_BTC=False,
PM='',
process_datafile_path='',
market_dynamic_labeling_visualization_paths=''
)