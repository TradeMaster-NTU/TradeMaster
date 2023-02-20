import json
import logging
import os
import os.path as osp
import pathlib
import subprocess
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import pytz
from flask import Flask, request, jsonify
from mmcv import Config

from trademaster.utils import replace_cfg_vals
from flask_cors import CORS

tz = pytz.timezone('Asia/Shanghai')

ROOT = str(pathlib.Path(__file__).resolve().parents[1])
sys.path.append(ROOT)

app = Flask(__name__)
CORS(app, resources={r"/TradeMaster/*": {"origins": "*"}})


def run_cmd(cmd):
    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    command_output = process.stdout.read().decode('utf-8')
    return command_output


def get_logger():
    logger = logging.getLogger('server')
    logger.setLevel(logging.DEBUG)

    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        '[%(asctime)s][%(thread)d][%(filename)s][line: %(lineno)d][%(levelname)s] ## %(message)s')
    ch.setFormatter(formatter)

    logger.addHandler(ch)

    return logger


logger = get_logger()
executor = ThreadPoolExecutor()


class Server():
    def __init__(self):
        self.logger = None
        self.sessions = self.load_sessions()

    def parameters(self):
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
                "order_excecution:PD_BTC": "2013-04-29",
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
            "style_test": [
                "bear_market",
                "bull_market",
                "oscillation_market"
            ]
        }
        return res

    def train_scripts(self, task_name, dataset_name, optimizer_name, loss_name, agent_name):
        if task_name == "algorithmic_trading":
            return os.path.join(ROOT, "tools", "algorithmic_trading", "train.py")
        elif task_name == "order_execution":
            if agent_name == "eteo":
                return os.path.join(ROOT, "tools", "order_execution", "train_eteo.py")
            elif agent_name == "pd":
                return os.path.join(ROOT, "tools", "order_execution", "train_pd.py")
        elif task_name == "portfolio_management":
            if dataset_name == "dj30":
                if agent_name == "deeptrader":
                    return os.path.join(ROOT, "tools", "portfolio_management", "train_deeptrader.py")
                elif agent_name == "eiie":
                    return os.path.join(ROOT, "tools", "portfolio_management", "train_eiie.py")
                elif agent_name == "investor_imitator":
                    return os.path.join(ROOT, "tools", "portfolio_management", "train_investor_imitator.py")
                elif agent_name == "sarl":
                    return os.path.join(ROOT, "tools", "portfolio_management", "train_sarl.py")
            elif dataset_name == "exchange":
                return os.path.join(ROOT, "tools", "portfolio_management", "train.py")

    def load_sessions(self):
        if os.path.exists("session.json"):
            with open("session.json", "r") as op:
                sessions = json.load(op)
        else:
            self.dump_sessions({})
            sessions = {}

        return sessions

    def dump_sessions(self, data):
        if os.path.exists("session.json"):
            with open("session.json", "r") as op:
                sessions = json.load(op)
        else:
            sessions = {}
        sessions.update(data)

        with open("session.json", "w") as op:
            json.dump(sessions, op)
        return sessions

    def get_parameters(self, request):
        logger.info("get parameters start.")
        param = self.parameters()
        logger.info("get parameters end.")
        return jsonify(param)

    def train(self, request):
        request_json = json.loads(request.get_data(as_text=True))
        try:
            task_name = request_json.get("task_name")
            dataset_name = request_json.get("dataset_name").split(":")[-1]
            optimizer_name = request_json.get("optimizer_name")
            loss_name = request_json.get("loss_name")
            agent_name = request_json.get("agent_name").split(":")[-1]
            start_date = request_json.get("start_date")
            end_date = request_json.get("end_date")
            session_id = str(uuid.uuid1())

            cfg_path = os.path.join(ROOT, "configs", task_name,
                                    f"{task_name}_{dataset_name}_{agent_name}_{agent_name}_{optimizer_name}_{loss_name}.py")
            train_script_path = self.train_scripts(task_name, dataset_name, optimizer_name, loss_name, agent_name)
            work_dir = os.path.join(ROOT, "work_dir", session_id,
                                    f"{task_name}_{dataset_name}_{agent_name}_{agent_name}_{optimizer_name}_{loss_name}")
            if not os.path.exists(work_dir):
                os.makedirs(work_dir)

            cfg = Config.fromfile(cfg_path)
            cfg = replace_cfg_vals(cfg)
            cfg.work_dir = "work_dir/{}/{}".format(session_id,
                                                   f"{task_name}_{dataset_name}_{agent_name}_{agent_name}_{optimizer_name}_{loss_name}")
            cfg.trainer.work_dir = cfg.work_dir

            # build dataset
            data = pd.read_csv(os.path.join(ROOT, cfg.data.data_path, "data.csv"), index_col=0)
            data = data[(data["date"] >= start_date) & (data["date"] < end_date)]

            indexs = range(len(data.index.unique()))

            train_indexs = indexs[:int(len(indexs) * 0.8)]
            val_indexs = indexs[int(len(indexs) * 0.8):int(len(indexs) * 0.9)]
            test_indexs = indexs[int(len(indexs) * 0.9):]

            train_data = data.loc[train_indexs, :]
            train_data.index = train_data.index - train_data.index.min()

            val_data = data.loc[val_indexs, :]
            val_data.index = val_data.index - val_data.index.min()

            test_data = data.loc[test_indexs, :]
            test_data.index = test_data.index - test_data.index.min()

            train_data.to_csv(os.path.join(work_dir, "train.csv"))
            cfg.data.train_path = "{}/{}".format(cfg.work_dir, "train.csv")
            val_data.to_csv(os.path.join(work_dir, "valid.csv"))
            cfg.data.valid_path = "{}/{}".format(cfg.work_dir, "valid.csv")
            test_data.to_csv(os.path.join(work_dir, "test.csv"))
            cfg.data.test_path = "{}/{}".format(cfg.work_dir, "test.csv")

            cfg_path = os.path.join(work_dir, osp.basename(cfg_path))
            cfg.dump(cfg_path)
            logger.info(cfg)

            log_path = os.path.join(work_dir, "train_log.txt")

            self.sessions = self.dump_sessions({session_id: {
                "work_dir": work_dir,
                "cfg_path": cfg_path,
                "script_path": train_script_path,
                "train_log_path": log_path,
                "test_log_path": os.path.join(os.path.dirname(log_path), "test_log.txt")
            }})

            cmd = "conda activate python3.9 && nohup python -u {} --config {} --task_name train > {} 2>&1 &".format(
                train_script_path,
                cfg_path,
                log_path)
            executor.submit(run_cmd, cmd)
            logger.info(cmd)

            error_code = 0
            info = "request success, start train"
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
            return jsonify(res)

    def train_status(self, request):
        request_json = json.loads(request.get_data(as_text=True))
        try:

            self.sessions = self.load_sessions()
            session_id = request_json.get("session_id")
            if session_id in self.sessions:
                cmd = "tail -n 2000 {}".format(self.sessions[session_id]["train_log_path"])
                info = run_cmd(cmd)
            else:
                info = "there are no train status"

            res = {
                "info": info,
            }
            logger.info("get train status success")
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

            self.sessions = self.load_sessions()
            session_id = request_json.get("session_id")

            work_dir = self.sessions[session_id]["work_dir"]
            script_path = self.sessions[session_id]["script_path"]
            cfg_path = self.sessions[session_id]["cfg_path"]
            log_path = self.sessions[session_id]["test_log_path"]

            cmd = "conda activate python3.9 && nohup python -u {} --config {} --task_name test > {} 2>&1 &".format(
                script_path,
                cfg_path,
                log_path)
            executor.submit(run_cmd, cmd)
            logger.info(cmd)

            error_code = 0
            info = "request success, start test"
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

    def test_status(self, request):
        request_json = json.loads(request.get_data(as_text=True))
        try:

            self.sessions = self.load_sessions()
            session_id = request_json.get("session_id")
            if session_id in self.sessions:
                cmd = "tail -n 2000 {}".format(self.sessions[session_id]["test_log_path"])
                info = run_cmd(cmd)
            else:
                info = "there are no test status"

            res = {
                "info": info,
            }
            logger.info("get train status success")
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

    def style_test(self, request):
        request_json = json.loads(request.get_data(as_text=True))
        try:

            session_id = request_json.get("session_id")
            style_test_mode = request_json.get("style_test")

            error_code = 0
            info = "style test request success"
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

@app.route("/api/TradeMaster/style_test", methods=["POST"])
def style_test():
    res = SERVER.style_test(request)
    return res


@app.route("/api/TradeMaster/healthcheck", methods=["GET"])
def health_check():
    res = HEALTHCHECK.run(request)
    return res


if __name__ == "__main__":
    host = "0.0.0.0"
    port = 8080
    app.run(host, port)
