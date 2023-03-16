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
        return cls._instance

pika_connection = PikaConnection().get_instance()