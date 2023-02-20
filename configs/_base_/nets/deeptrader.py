
act_net = dict(
    type = "AssetScoringNet",
    N = None,
    K_l = None,
    num_inputs = None,
    num_channels=[12, 12, 12],
    kernel_size = 2,
    dropout = 0.2
)

cri_net = dict(
    type = "AssetScoringValueNet",
    N = None,
    K_l = None,
    num_inputs = None,
    num_channels=[12, 12, 12],
    kernel_size = 2,
    dropout = 0.2
)

market_net = dict(
    type = "MarketScoringNet",
    n_features = None,
    hidden_size = 12
)

