
act = dict(
    type = "EIIEConv",
    input_dim = None,
    output_dim=1,
    time_steps=None,
    kernel_size=3,
    dims = [32]
)

cri = dict(
    type = "EIIECritic",
    input_dim = None,
    action_dim = None,
    output_dim=1,
    time_steps=None,
    num_layers = 1,
    hidden_size=32
)