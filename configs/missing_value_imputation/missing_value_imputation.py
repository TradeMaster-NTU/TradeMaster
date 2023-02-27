data = dict(
    type='CSDI_Imputation',
    eval_length=10,
    seed=1,
    batch_size=16,
    missing_ratio=0.1,
    nsample = 100,
    dataset_name = "dj30",
    tic_name = "IBM",
    epochs = 200,
    lr = 1.0e-3,
    diffusion = dict(
    layers = 4,
    channels = 64,
    nheads = 8,
    diffusion_embedding_dim = 128,
    beta_start = 0.0001,
    beta_end = 0.5,
    num_steps = 50,
    schedule = "quad"),
    model = dict(
    is_unconditional = 0,
    timeemb = 128,
    featureemb = 16,
    target_strategy = "random"))


