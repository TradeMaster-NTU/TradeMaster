
data = dict(
    type = "PortfolioManagementDataset",
    data_path = "data/portfolio_management/exchange",
    train_path = "data/portfolio_management/exchange/train.csv",
    valid_path = "data/portfolio_management/exchange/valid.csv",
    test_path = "data/portfolio_management/exchange/test.csv",
    tech_indicator_list = [
        "zopen",
        "zhigh",
        "zlow",
        "zadjcp",
        "zclose",
        "zd_5",
        "zd_10",
        "zd_15",
        "zd_20",
        "zd_25",
        "zd_30"
    ],
    initial_amount = 100000,
    transaction_cost_pct = 0.001
)