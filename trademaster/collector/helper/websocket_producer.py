import json
import websocket
from celery_app import app
from pika_connection import pika_connection

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
        "timestep": timestep,
        "bid1_price": bid1_price,
        "bid1_size": bid1_size,
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
        "timestep": timestep,
        "open": open,
        "high": high,
        "low": low,
        "close": close
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
            store_data_in_rabbitmq(combined_data)

        # reset orderbook_data and kline_data
        orderbook_data = None
        kline_data = None

def store_data_in_rabbitmq(combined_data):
    channel = pika_connection.channel()

    # init queue
    channel.queue_declare(queue='binance_data')

    # to json string
    message = json.dumps(combined_data)

    # push a message
    channel.basic_publish(exchange='',
                          routing_key='binance_data',
                          body=message)
    print("Stored data in RabbitMQ:")
    print(message)

def start_producer():
    ws = websocket.WebSocketApp(
        "wss://stream.binance.com:9443/stream?streams=btcusdt@kline_1s/btcusdt@depth@1000ms",
        on_open=on_open,
        on_error=on_error,
        on_close=on_close,
        on_message=on_message
    )
    ws.run_forever()

@app.task
def restart_producer():
    start_producer()

if __name__ == "__main__":
    start_producer()