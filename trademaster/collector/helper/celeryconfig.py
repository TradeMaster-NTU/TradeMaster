from datetime import timedelta

broker_url = 'pyamqp://guest@localhost//'
result_backend = 'rpc://'

timezone = 'UTC'

beat_schedule = {
    'fetch-and-store-data-every-24-hours': {
        'task': 'tasks.fetch_and_store_data',
        'schedule': timedelta(hours=24),
    },
}