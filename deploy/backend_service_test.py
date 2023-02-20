import os
import sys
import uuid
from flask import Flask, request, jsonify
import time
import logging
import pytz
import json

tz = pytz.timezone('Asia/Shanghai')

root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, root)
app = Flask(__name__)


def logger():
    logger = logging.getLogger("server")
    logger.setLevel(level=logging.INFO)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    if console not in logger.handlers:
        logger.addHandler(console)
    return logger


logger = logger()


class Server():
    def __init__(self, debug=True):
        if debug:
            self.debug()
        pass

    def debug(self):
        pass

    def get_parameters(self, request):
        logger.info("get_parameters start.")
        res = {
            "task_name": ["algorithmic_trading", "order_execution", "portfolio_management"],
            "dataset_name": ["algorithmic_trading:BTC",
                             "order_excecution:BTC",
                             "portfolio_management:dj30",
                             "portfolio_management:exchange"],
            "optimizer_name": ["adam", "adaw"],
            "loss_name": ["mae", "mse"],
            "agent_name": [
                "algorithmic_trading:dqn",
                "order_execution:eteo",
                "order_execution:pd",
                "portfolio_management:a2c",
                "portfolio_management:ddpg",
                "portfolio_management:deeptrader",
                "portfolio_management:eiie",
                "portfolio_management:investor_imitator",
                "portfolio_management:pg",
                "portfolio_management:ppo",
                "portfolio_management:sac",
                "portfolio_management:sarl",
                "portfolio_management:td3"
            ]
        }
        logger.info("get_parameters end.")
        return jsonify(res)


    def start(self, request):
        request_json = json.loads(request.get_data(as_text=True))
        try:
            task_name = request_json.get("task_name")
            dataset_name = request_json.get("dataset_name")
            optimizer_name = request_json.get("optimizer_name")
            loss_name = request_json.get("loss_name")
            agent_name = request_json.get("agent_name")

            session_id = str(uuid.uuid1())

            error_code = 0
            info = "request success"
            res = {
                "error_code": error_code,
                "info": info,
                "session_id": session_id
            }
            logger.info(info)
            return jsonify(res)

        except Exception as e:
            error_code = 1
            info = "request data error, {}".format(e)
            res = {
                "error_code": error_code,
                "info": info,
                "session_id": ""
            }
            logger.info(info)
            return jsonify(res)

    def start_status(self, request):
        request_json = json.loads(request.get_data(as_text=True))
        try:

            session_id = request_json.get("session_id")

            error_code = 0
            info = "test for start status"
            res = {
                "info": info,
            }
            logger.info(info)
            return jsonify(res)

        except Exception as e:
            error_code = 1
            info = "request data error, {}".format(e)
            res = {
                "error_code": error_code,
                "info": info,
                "session_id": ""
            }
            logger.info(info)
            return jsonify(res)

    def test(self, request):
        request_json = json.loads(request.get_data(as_text=True))
        try:

            session_id = request_json.get("session_id")

            error_code = 0
            info = "request success"
            res = {
                "error_code": error_code,
                "info": info,
                "session_id": session_id
            }
            logger.info(info)
            return jsonify(res)

        except Exception as e:
            error_code = 1
            info = "request data error, {}".format(e)
            res = {
                "error_code": error_code,
                "info": info,
                "session_id": session_id
            }
            logger.info(info)
            return jsonify(res)

    def test_status(self, request):
        request_json = json.loads(request.get_data(as_text=True))
        try:

            session_id = request_json.get("session_id")

            error_code = 0
            info = "test for test status"
            res = {
                "info": info,
            }
            logger.info(info)
            return jsonify(res)

        except Exception as e:
            error_code = 1
            info = "request data error, {}".format(e)
            res = {
                "error_code": error_code,
                "info": info,
                "session_id": ""
            }
            logger.info(info)
            return jsonify(res)


class HealthCheck():
    def __init__(self):
        super(HealthCheck, self).__init__()

    def run(self, request):
        start = time.time()
        if request.method == "GET":
            logger.info("health check start.")
            error_code = 0
            info = "health check"
            time_consuming = (time.time() - start) * 1000
            res = {
                "data": {},
                "error_code": error_code,
                "info": info,
                "time_consuming": time_consuming
            }
            logger.info("health check end.")
            return jsonify(res)

SERVER = Server()
HEALTHCHECK = HealthCheck()

@app.route("/TradeMaster/getParameters", methods=["GET"])
def getParameters():
    res = SERVER.get_parameters(request)
    return res

@app.route("/TradeMaster/start", methods=["POST"])
def start():
    res = SERVER.start(request)
    return res

@app.route("/TradeMaster/start_status", methods=["POST"])
def start_status():
    res = SERVER.start_status(request)
    return res

@app.route("/TradeMaster/test", methods=["POST"])
def test():
    res = SERVER.start(request)
    return res

@app.route("/TradeMaster/test_status", methods=["POST"])
def test_status():
    res = SERVER.start_status(request)
    return res

@app.route("/TradeMaster/healthcheck", methods=["GET"])
def health_check():
    res = HEALTHCHECK.run(request)
    return res


if __name__ == "__main__":
    host = "0.0.0.0"
    port = 8080
    app.run(host, port)
