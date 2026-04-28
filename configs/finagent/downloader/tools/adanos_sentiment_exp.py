root = None
workdir = "workdir"
tag = "adanos_sentiment_exp"
batch_size = 1

downloader = dict(
    type = "AdanosSentimentDownloader",
    root = root,
    token = None,
    base_url = "https://api.adanos.org",
    start_date = "2023-01-01",
    end_date = "2024-01-01",
    delay = 0.5,
    days = 30,
    sources = ["reddit", "x", "news", "polymarket"],
    timeout = 30,
    stocks_path = "configs/_stock_list_/exp_stocks.txt",
    workdir = workdir,
    tag = tag,
)
