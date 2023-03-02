task_name = "high_frequency_trading"
dataset_name = "BTC"
optimizer_name = "adam"
loss_name = "mse"
auxiliry_loss_name = "KLdiv"
net_name = "high_frequency_trading_dqn"
agent_name = "ddqn"
work_dir = f"work_dir/{task_name}_{dataset_name}_{net_name}_{agent_name}_{optimizer_name}_{loss_name}"

_base_ = [
    f"../_base_/datasets/{task_name}/{dataset_name}.py",
    f"../_base_/environments/{task_name}/env.py",
    f"../_base_/agents/{task_name}/{agent_name}.py",
    f"../_base_/trainers/{task_name}/trainer.py",
    f"../_base_/losses/{loss_name}.py",
    f"../_base_/optimizers/{optimizer_name}.py",
    f"../_base_/nets/{net_name}.py",
]

batch_size = 512
data = dict(
    type="HighFrequencyTradingDataset",
    data_path="data/high_frequency_trading/small_BTC",
    train_path="data/high_frequency_trading/small_BTC/train.csv",
    valid_path="data/high_frequency_trading/small_BTC/valid.csv",
    test_path="data/high_frequency_trading/small_BTC/test.csv",
    test_dynamic_path="data/high_frequency_trading/small_BTC/test_labeled_3_300_-0.01_0.01.csv",
    tech_indicator_list=[
        "imblance_volume_oe",
        "sell_spread_oe",
        "buy_spread_oe",
        "kmid2",
        "bid1_size_n",
        "ksft2",
        "ma_10",
        "ksft",
        "kmid",
        "ask1_size_n",
        "trade_diff",
        "qtlu_10",
        "qtld_10",
        "cntd_10",
        "beta_10",
        "roc_10",
        "bid5_size_n",
        "rsv_10",
        "imxd_10",
        "ask5_size_n",
        "ma_30",
        "max_10",
        "qtlu_30",
        "imax_10",
        "imin_10",
        "min_10",
        "qtld_30",
        "cntn_10",
        "rsv_30",
        "cntp_10",
        "ma_60",
        "max_30",
        "qtlu_60",
        "qtld_60",
        "cntd_30",
        "roc_30",
        "beta_30",
        "bid4_size_n",
        "rsv_60",
        "ask4_size_n",
        "imxd_30",
        "min_30",
        "max_60",
        "imax_30",
        "imin_30",
        "cntd_60",
        "roc_60",
        "beta_60",
        "cntn_30",
        "min_60",
        "cntp_30",
        "bid3_size_n",
        "imxd_60",
        "ask3_size_n",
        "sell_volume_oe",
        "imax_60",
        "imin_60",
        "cntn_60",
        "buy_volume_oe",
        "cntp_60",
        "bid2_size_n",
        "kup",
        "bid1_size",
        "ask1_size",
        "std_30",
        "ask2_size_n",
    ],
    transcation_cost=0,
    backward_num_timestamp=1,
    max_holding_number=0.01,
    num_action=11,
    max_punish=1e12,
    episode_length=14400,
)

environment = dict(type="HighFrequencyTradingEnvironment")
train_environment = dict(type="HighFrequencyTradingTrainingEnvironment")

agent = dict(
    type="HighFrequencyTradingDDQN",
    auxiliary_coffient=512,
    reward_scale=2**0,
    repeat_times=1,
    gamma=0.99,
    batch_size=64,
    clip_grad_norm=3.0,
    soft_update_tau=0,
    state_value_tau=5e-3,
)
trainer = dict(
    type="HighFrequencyTradingTrainer",
    epochs=10,
    work_dir=work_dir,
    seeds=12345,
    batch_size=512,
    horizon_len=512,
    buffer_size=1e5,
    num_threads=8,
    if_remove=False,
    if_discrete=True,
    if_off_policy=True,
    if_keep_save=True,
    if_over_write=False,
    if_save_buffer=False,
)
loss = dict(type="HFTLoss", ada=1)
optimizer = dict(type="Adam", lr=0.001)
act = dict(type="HFTQNet", state_dim=66, action_dim=11, dims=16, explore_rate=0.01)
cri = None