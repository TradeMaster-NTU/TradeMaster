from datetime import timedelta


broker_url = 'amqp://zwt:123456@localhost/vhost'
result_backend = 'rpc://'

timezone = 'UTC'

beat_schedule = {
    'fetch-and-store-data-every-24-hours': {
        'task': 'tasks.fetch_and_store_data',
        'schedule': timedelta(hours=24),
    },
}