import sys
from pathlib import Path
import websocket
import json
import threading

ROOT = str(Path(__file__).resolve().parents[2])
sys.path.append(ROOT)

from trademaster.collector.builder import COLLECTORS
from trademaster.collector.custom import CollectorBase

def on_open(ws):
    print("WebSocket connection opened")

def on_error(ws, error):
    print(f"WebSocket error: {error}")

def on_close(ws):
    print("WebSocket connection closed")

def on_message(ws, message):
    data = json.loads(message)["data"]

    if 'e' in data:
        event_type = data['e']
        if event_type == 'depthUpdate':
            print("Orderbook data received")
            print(data)
        elif event_type == 'kline':
            print("Kline data received")
            print(data)

def start_websocket():
    ws = websocket.WebSocketApp(
        "wss://stream.binance.com:9443/stream?streams=btcusdt@kline_1s/btcusdt@depth@1000ms",
        on_open=on_open,
        on_error=on_error,
        on_close=on_close,
        on_message=on_message
    )
    ws.run_forever()

@COLLECTORS.register_module()
class BinanceRealTimeDataCollector(CollectorBase):
    def __init__(self):
        super(BinanceRealTimeDataCollector, self).__init__()

    def run(self):
        pass

if __name__ == "__main__":
    websocket_thread = threading.Thread(target=start_websocket)
    websocket_thread.start()