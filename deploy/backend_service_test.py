import os
import sys
import uuid
from flask import Flask, request, jsonify
import time
import logging
import pytz
import json
from flask_cors import CORS
import base64

tz = pytz.timezone('Asia/Shanghai')

root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, root)
app = Flask(__name__)
CORS(app, resources={r"/TradeMaster/*": {"origins": "*"}})

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
                             "order_excecution:PD_BTC",
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
            ],
            "start_date": {
                "algorithmic_trading:BTC": "2013-04-29",
                "order_excecution:BTC": "2021-04-07",
                "order_excecution:PD_BTC":"2013-04-29",
                "portfolio_management:dj30": "2012-01-04",
                "portfolio_management:exchange": "2000-01-27",
            },
            "end_date": {
                "algorithmic_trading:BTC": "2021-07-05",
                "order_excecution:BTC": "2021-04-19",
                "order_excecution:PD_BTC": "2021-07-05",
                "portfolio_management:dj30": "2021-12-31",
                "portfolio_management:exchange": "2019-12-31",
            },
            "dynamics_test":[
                "bear_market",
                "bull_market",
                "oscillation_market"
            ]
        }
        logger.info("get_parameters end.")
        return jsonify(res)

    def evaluation_parameters(self, request):
        request_json = json.loads(request.get_data(as_text=True))
        session_id = request_json.get("session_id")
        self.sessions = self.load_sessions()
        if session_id in self.sessions:
            dataset = self.sessions[session_id]["dataset"]
        res = {
            "dataset_name": [dataset],
            "start_date": {
                "algorithmic_trading:BTC": "2020-03-03",
                "algorithmic_trading:FX": "2017-12-22",
                "order_excecution:BTC": "2021-04-17",
                "order_excecution:PD_BTC": "2020-09-10",
                "portfolio_management:dj30": "2021-04-01",
                "portfolio_management:exchange": "2019-08-09",
            },
            "end_date": {
                "algorithmic_trading:BTC": "2021-07-05",
                "algorithmic_trading:FX": "2019-12-31",
                "order_excecution:BTC": "2021-04-19",
                "order_excecution:PD_BTC": "2021-07-05",
                "portfolio_management:dj30": "2021-12-31",
                "portfolio_management:exchange": "2019-12-31",
            },
            "number_of_market_style": ["3"],
            "length_time_slice": {
                "algorithmic_trading:BTC": "12",
                "algorithmic_trading:FX": "24",
                "order_excecution:BTC": "32",
                "order_excecution:PD_BTC": "24",
                "portfolio_management:dj30": "24",
                "portfolio_management:exchange": "24"
            },
            "bear_threshold": {
                "algorithmic_trading:BTC": "-0.4",
                "algorithmic_trading:FX": "-0.05",
                "order_excecution:BTC": "-0.01",
                "order_excecution:PD_BTC": "-0.15",
                "portfolio_management:dj30": "-0.25",
                "portfolio_management:exchange": "-0.05"
            },
            "bull_threshold": {
                "algorithmic_trading:BTC": "0.4",
                "algorithmic_trading:FX": "0.05",
                "order_excecution:BTC": "0.01",
                "order_excecution:PD_BTC": "0.15",
                "portfolio_management:dj30": "0.25",
                "portfolio_management:exchange": "0.05"
            }
        }
        return res
    def train(self, request):
        request_json = json.loads(request.get_data(as_text=True))
        try:
            task_name = request_json.get("task_name")
            dataset_name = request_json.get("dataset_name")
            optimizer_name = request_json.get("optimizer_name")
            loss_name = request_json.get("loss_name")
            agent_name = request_json.get("agent_name")
            start_date = request_json.get("start_date")
            end_date = request_json.get("end_date")

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

    def train_status(self, request):
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

    def start_market_dynamics_labeling(self, request):
        request_json = json.loads(request.get_data(as_text=True))
        try:
            # market_dynamics_labeling parameters
            args={}
            args['dataset_name'] = request_json.get("dynamics_test_dataset_name")
            args['number_of_market_dynamics'] = request_json.get("number_of_market_style")
            if args['number_of_market_dynamics'] not in [3,4]:
                raise Exception('only support dynamics number of 3 or 4 for now')
            args['minimun_length'] = request_json.get("minimun_length")
            args['Granularity'] = request_json.get("Granularity")
            args['bear_threshold'] = request_json.get("bear_threshold")
            args['bull_threshold'] = request_json.get("bull_threshold")
            args['task_name']=request_json.get("task_name")

            #TODO: assign a session id if no session id is provided

            session_id = request_json.get("session_id")




            #fake respone message
            with open('Market_dynmacis_labeling.png',
                      "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read())

            error_code = 0
            info = "request success, show market dynamics labeling visualization"

            res = {
                "error_code": error_code,
                "info": info,
                "market_dynamic_labeling_visulization": encoded_string
            }
            logger.info(info)
            return jsonify(res)

        except Exception as e:
            error_code = 1
            info = "request data error, {}".format(e)
            res = {
                "error_code": error_code,
                "info": info,
                "market_dynamic_labeling_visulization": ""
            }
            logger.info(info)
            return jsonify(res)

    def save_market_dynamics_labeling(self, request):
        request_json = json.loads(request.get_data(as_text=True))
        try:
            session_id = request_json.get("session_id")



            error_code = 0
            info = "request success, save market dynamics"
            res = {
                "error_code": error_code,
                "info": info
            }
            logger.info(info)
            return jsonify(res)

        except Exception as e:
            error_code = 1
            info = "request data error, {}".format(e)
            res = {
                "error_code": error_code,
                "info": info
            }
            logger.info(info)
            return jsonify(res)


    def run_dynamics_test(self, request):
        request_json = json.loads(request.get_data(as_text=True))
        try:
            #
            dynamics_test_label = request_json.get("test_dynamic_label")
            session_id = request_json.get("session_id")




            with open('Radar_plot.png', "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read())

            #print log output
            dynamics_test_log_info = 'test_log_placeholder'


            error_code = 0
            info = f"request success, start test market {dynamics_test_label}\n\n"
            res = {
                "error_code": error_code,
                "info": info+dynamics_test_log_info,
                "session_id": session_id,
                'radar_plot':encoded_string
            }
            logger.info(info)
            return jsonify(res)

        except Exception as e:
            error_code = 1
            info = "request data error, {}".format(e)
            res = {
                "error_code": error_code,
                "info": info,
                "session_id": "",
                'radar_plot':""
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

@app.route("/api/TradeMaster/getParameters", methods=["GET"])
def getParameters():
    res = SERVER.get_parameters(request)
    return res
@app.route("/api/TradeMaster/evaluation_getParameters", methods=["GET"])
def evaluation_getParameters():
    res = SERVER.evaluation_parameters(request)
    return res


@app.route("/api/TradeMaster/train", methods=["POST"])
def train():
    res = SERVER.train(request)
    return res


@app.route("/api/TradeMaster/train_status", methods=["POST"])
def train_status():
    res = SERVER.train_status(request)
    return res


@app.route("/api/TradeMaster/test", methods=["POST"])
def test():
    res = SERVER.test(request)
    return res


@app.route("/api/TradeMaster/test_status", methods=["POST"])
def test_status():
    res = SERVER.test_status(request)
    return res


@app.route("/api/TradeMaster/start_market_dynamics_labeling", methods=["POST"])
def start_market_dynamics_labeling():
    res = SERVER.start_market_dynamics_labeling(request)
    return res


@app.route("/api/TradeMaster/save_market_dynamics_labeling", methods=["POST"])
def save_market_dynamics_labeling():
    res = SERVER.save_market_dynamics_labeling(request)
    return res


@app.route("/api/TradeMaster/run_dynamics_test", methods=["POST"])
def run_style_test():
    res = SERVER.run_dynamics_test(request)
    return res


@app.route("/api/TradeMaster/healthcheck", methods=["GET"])
def health_check():
    res = HEALTHCHECK.run(request)
    return res


if __name__ == "__main__":
    host = "0.0.0.0"
    port = 8080
    app.run(host, port)
