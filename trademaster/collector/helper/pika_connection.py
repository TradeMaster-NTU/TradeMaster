import pika

class PikaConnection:
    _instance = None

    def __init__(self):
        if not PikaConnection._instance:
            print("__init__ method called but nothing is created")
        else:
            print("Instance already created:", self.get_instance())

    @classmethod
    def get_instance(cls):
        if not cls._instance:
            cls._instance = pika.BlockingConnection(
                pika.ConnectionParameters(host='localhost')
            )

            channel = cls._instance.channel()

            channel.exchange_declare(exchange="binance_data_dead_letter_exchange", exchange_type='direct')
            channel.queue_declare(queue="binance_data")
            dead_letter_queue = channel.queue_declare(queue='binance_data_dead_letter',).method.queue
            channel.queue_bind(exchange="binance_data_dead_letter_exchange", queue=dead_letter_queue)

        return cls._instance

pika_connection = PikaConnection().get_instance()