from celery import Celery

app = Celery('binance_data')
app.config_from_object('celery_config')