import pika

class RabbitMQConnection:
    _instance = None

    @staticmethod
    def get_instance():
        if RabbitMQConnection._instance is None:
            RabbitMQConnection._instance = pika.BlockingConnection(pika.ConnectionParameters('localhost'))
        return RabbitMQConnection._instance