
agent = dict(
    type = "AlgorithmicTradingDQN",
    max_step = 12345,
    reward_scale = 2**0,
    repeat_times = 1,
    gamma = 0.99,
    batch_size = 64,
    clip_grad_norm = 3.0,
    soft_update_tau = 0,
    state_value_tau = 5e-3
)