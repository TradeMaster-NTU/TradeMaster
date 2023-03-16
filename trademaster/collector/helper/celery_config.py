from datetime import timedelta

broker_url = 'amqp://zwt:123456@localhost/vhost'
result_backend = 'rpc://'
timezone = 'UTC'

beat_schedule = {
    'restart-producer-every-24-hours': {
        'task': 'websocket_producer.restart_producer',
        'schedule': timedelta(hours=24),
    },
}

imports = ('websocket_producer', 'websocket_consumer')