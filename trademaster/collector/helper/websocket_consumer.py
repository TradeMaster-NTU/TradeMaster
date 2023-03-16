from celery import shared_task
from pika_connection import pika_connection
import json

@shared_task
def consume_data():
    channel = pika_connection.channel()
    channel.queue_declare(queue='binance_data')

    def callback(ch, method, properties, body):
        data = json.loads(body)
        print("Received data from RabbitMQ:")
        ch.basic_ack(delivery_tag=method.delivery_tag)
        return data

    while True:
        method_frame, header_frame, body = channel.basic_get(queue='binance_data', auto_ack=False)
        if method_frame:
            data = callback(channel, method_frame, header_frame, body)
            yield data
        else:
            break

if __name__ == "__main__":
    data_generator = consume_data()

    for data in data_generator:
        print(data)