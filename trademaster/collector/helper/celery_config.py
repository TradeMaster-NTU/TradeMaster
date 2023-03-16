from datetime import timedelta

broker_url = 'amqp://zwt:123456@localhost/vhost'
result_backend = 'rpc://'
timezone = 'UTC'

imports = ('websocket_producer', 'websocket_consumer')