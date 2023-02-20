
act = dict(
    type="ETEOStacked",
    dims=[128,128],
    time_steps=10,
    action_dim=2,
    state_dim =10,
    explore_rate=0.25
)

cri = dict(
    type="ETEOStacked",
    dims=[128,128],
    time_steps=10,
    action_dim=2,
    state_dim =10,
    explore_rate=0.25
)