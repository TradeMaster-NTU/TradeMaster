import websocket
import threading
import json
from celery import Celery
from rabbitmq_connection import RabbitMQConnection

app = Celery('tasks', broker='amqp://zwt:123456@localhost/vhost')

orderbook_data = None
kline_data = None

def process_orderbook_data(data):
    timestep = data["E"]
    bid = data["b"]
    bid1_price, bid1_size = bid[0]
    bid2_price, bid2_size = bid[1]
    bid3_price, bid3_size = bid[2]
    bid4_price, bid4_size = bid[3]
    bid5_price, bid5_size = bid[4]

    ask = data["a"]
    ask1_price, ask1_size = ask[0]
    ask2_price, ask2_size = ask[1]
    ask3_price, ask3_size = ask[2]
    ask4_price, ask4_size = ask[3]
    ask5_price, ask5_size = ask[4]
    
    data = {
        "timestep":timestep,
        "bid1_price":bid1_price,
        "bid1_size":bid1_size,
        "bid2_price": bid2_price,
        "bid2_size": bid2_size,
        "bid3_price": bid3_price,
        "bid3_size": bid3_size,
        "bid4_price": bid4_price,
        "bid4_size": bid4_size,
        "bid5_price": bid5_price,
        "bid5_size": bid5_size,
        "ask1_price": ask1_price,
        "ask1_size": ask1_size,
        "ask2_price": ask2_price,
        "ask2_size": ask2_size,
        "ask3_price": ask3_price,
        "ask3_size": ask3_size,
        "ask4_price": ask4_price,
        "ask4_size": ask4_size,
        "ask5_price": ask5_price,
        "ask5_size": ask5_size,
    }
    return json.dumps(data)

def process_kline_data(data):
    timestep = data["E"]
    kdata = data["k"]
    open = kdata["o"]
    high = kdata["h"]
    low = kdata["l"]
    close = kdata["c"]

    data = {
        "timestep":timestep,
        "open":open,
        "high":high,
        "low":low,
        "close":close
    }
    return json.dumps(data)

def on_open(ws):
    print("WebSocket connection opened")

def on_error(ws, error):
    print(f"WebSocket error: {error}")

def on_close(ws):
    print("WebSocket connection closed")

def on_message(ws, message):
    global orderbook_data, kline_data
    data = json.loads(message)["data"]

    if 'e' in data:
        event_type = data['e']
        if event_type == 'depthUpdate':
            orderbook_data = data
        elif event_type == 'kline':
            kline_data = data

    if orderbook_data and kline_data:
        if orderbook_data['E'] // 1000 == kline_data['E'] // 1000:
            combined_data = {
                'orderbook': process_orderbook_data(orderbook_data),
                'kline': process_kline_data(kline_data)
            }
            store_data_in_rabbitmq.delay(combined_data)

        # reset orderbook_data and kline_data
        orderbook_data = None
        kline_data = None

def start_websocket():
    ws = websocket.WebSocketApp(
        "wss://stream.binance.com:9443/stream?streams=btcusdt@kline_1s/btcusdt@depth@1000ms",
        on_open=on_open,
        on_error=on_error,
        on_close=on_close,
        on_message=on_message
    )
    ws.run_forever()

@app.task
def fetch_and_store_data():
    websocket_thread = threading.Thread(target=start_websocket)
    websocket_thread.start()

@app.task(queue='producer_queue')
def store_data_in_rabbitmq(combined_data):
    connection = RabbitMQConnection.get_instance()
    channel = connection.channel()

    # init queue
    channel.queue_declare(queue='combined_data_queue')

    # to json string
    message = json.dumps(combined_data)

    # push a message
    channel.basic_publish(exchange='',
                          routing_key='combined_data_queue',
                          body=message)
    print("Stored data in RabbitMQ:")
    print(combined_data)
@app.task(queue='consumer_queue')
def read_data_from_rabbitmq():
    connection = RabbitMQConnection.get_instance()
    channel = connection.channel()
    channel.queue_declare(queue='combined_data_queue')

    def callback(ch, method, properties, body):
        data = json.loads(body)
        print("Received data from RabbitMQ:")
        ch.basic_ack(delivery_tag=method.delivery_tag)
        return data

    while True:
        method_frame, header_frame, body = channel.basic_get(queue='combined_data_queue', auto_ack=False)
        if method_frame:
            data = callback(channel, method_frame, header_frame, body)
            yield data
        else:
            break

# if __name__ == "__main__":
#     data_generator = read_data_from_rabbitmq()
#
#     for data in data_generator:
#         print("Received data from RabbitMQ:")
#         print(data)