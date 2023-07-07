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
from flask import Flask, request, jsonify, Response
from mmcv import Config

ROOT = str(pathlib.Path(__file__).resolve().parents[1])
sys.path.append(ROOT)

from trademaster.utils import replace_cfg_vals, MRL_F2B_args_converter
from flask_cors import CORS
import os.path as osp
import pickle
import csv
import regex as re
import copy
from tools import market_dynamics_labeling
import base64

tz = pytz.timezone('Asia/Shanghai')

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
        self.evaluation_parameters_dict ={
            "dataset_name": [
                "algorithmic_trading:BTC",
                "algorithmic_trading:FX",
                # "order_execution:BTC",
                # "order_execution:PD_BTC",
                "portfolio_management:dj30",
                "portfolio_management:exchange",
            'market_dynamics_modeling:second_level_BTC_LOB'],
            "start_date": {
                "algorithmic_trading:BTC": "2015-10-01",
                "algorithmic_trading:FX": "2000-01-01",
                "order_execution:BTC": "2021-04-07",
                "order_execution:PD_BTC": "2013-04-29",
                "portfolio_management:dj30": "2012-01-04",
                "portfolio_management:exchange": "2000-01-27",
                'market_dynamics_modeling:second_level_BTC_LOB': "2020-09-01"
            },
            "end_date": {
                "algorithmic_trading:BTC": "2021-07-05",
                "algorithmic_trading:FX": "2019-12-31",
                "order_execution:BTC": "2021-04-19",
                "order_execution:PD_BTC": "2021-07-05",
                "portfolio_management:dj30": "2021-12-31",
                "portfolio_management:exchange": "2019-12-31",
                'market_dynamics_modeling:second_level_BTC_LOB': "2020-09-01"
            },
            "number_of_market_style": ["5"],
            "min_length_limit": {
                "algorithmic_trading:BTC": "24",
                "algorithmic_trading:FX": "24",
                "order_execution:BTC": "24",
                "order_execution:PD_BTC": "24",
                "portfolio_management:dj30": "24",
                "portfolio_management:exchange": "24",
                'market_dynamics_modeling:second_level_BTC_LOB': "60"
            },
            "filter_strength": {
                "algorithmic_trading:BTC": "1",
                "algorithmic_trading:FX": "1",
                "order_execution:BTC": "1",
                "order_execution:PD_BTC": "1",
                "portfolio_management:dj30": "1",
                "portfolio_management:exchange": "1",
                'market_dynamics_modeling:second_level_BTC_LOB': "1"
            },
            "max_length_expectation": {
                "algorithmic_trading:BTC": "144",
                "algorithmic_trading:FX": "144",
                "order_execution:BTC": "144",
                "order_execution:PD_BTC": "144",
                "portfolio_management:dj30": "144",
                "portfolio_management:exchange": "144",
                'market_dynamics_modeling:second_level_BTC_LOB': "3600"
            },
            "key_indicator": {
                "algorithmic_trading:BTC": "adjcp",
                "algorithmic_trading:FX": "adjcp",
                "order_execution:BTC": "adjcp",
                "order_execution:PD_BTC": "adjcp",
                "portfolio_management:dj30": "adjcp",
                "portfolio_management:exchange": "adjcp",
                'market_dynamics_modeling:second_level_BTC_LOB': "bid1_price"
            },
            "timestamp": {
                "algorithmic_trading:BTC": "date",
                "algorithmic_trading:FX": "date",
                "order_execution:BTC": "date",
                "order_execution:PD_BTC": "date",
                "portfolio_management:dj30": "date",
                "portfolio_management:exchange": "date",
                'market_dynamics_modeling:second_level_BTC_LOB': "timestamp"
            },
            "tic": {
                "algorithmic_trading:BTC": "data",
                "algorithmic_trading:FX": "data",
                "order_execution:BTC": "data",
                "order_execution:PD_BTC": "data",
                "portfolio_management:dj30": "tic",
                "portfolio_management:exchange": "tic",
                'market_dynamics_modeling:second_level_BTC_LOB': "BTCUSDT"
            },
            "labeling_method": {
                "algorithmic_trading:BTC": "quantile",
                "algorithmic_trading:FX": "quantile",
                "order_execution:BTC": "quantile",
                "order_execution:PD_BTC": "quantile",
                "portfolio_management:dj30": "quantile",
                "portfolio_management:exchange": "quantile",
                'market_dynamics_modeling:second_level_BTC_LOB': "quantile"
            },
            "merging_metric": {
                "algorithmic_trading:BTC": "DTW_distance",
                "algorithmic_trading:FX": "DTW_distance",
                "order_execution:BTC": "DTW_distance",
                "order_execution:PD_BTC": "DTW_distance",
                "portfolio_management:dj30": "DTW_distance",
                "portfolio_management:exchange": "DTW_distance",
                'market_dynamics_modeling:second_level_BTC_LOB': "DTW_distance"
            },
            "merging_threshold": {
                "algorithmic_trading:BTC": "0.03",
                "algorithmic_trading:FX": "0.03",
                "order_execution:BTC": "0.03",
                "order_execution:PD_BTC": "0.03",
                "portfolio_management:dj30": "0.03",
                "portfolio_management:exchange": "0.03",
                'market_dynamics_modeling:second_level_BTC_LOB': "0.0003"
            },
            "merging_dynamic_constraint": {
                "algorithmic_trading:BTC": 1,
                "algorithmic_trading:FX": 1,
                "order_execution:BTC": 1,
                "order_execution:PD_BTC": 1,
                "portfolio_management:dj30": 1,
                "portfolio_management:exchange": 1,
                'market_dynamics_modeling:second_level_BTC_LOB': 1
            },
            # TODO: get the number of dynamic parameters from session
            "test_dynamic_number": [
                "0",
                "1",
                "2",
                "3"
            ]
        }
        self.parameters = {
            "task_name": ["algorithmic_trading", "order_execution", "portfolio_management"],
            "dataset_name": [
                "algorithmic_trading:BTC",
                "algorithmic_trading:FX",
                "algorithmic_trading:JPM",
                "order_execution:BTC",
                "order_execution:PD_BTC",
                "portfolio_management:dj30",
                "portfolio_management:exchange"],
            "optimizer_name": ["adam"],
            "loss_name": ["mse"],
            "agent_name": [
                "algorithmic_trading:deepscalper",
                "order_execution:eteo",
                "order_execution:pd",
                "portfolio_management:a2c",
                "portfolio_management:ddpg",
                "portfolio_management:eiie",
                "portfolio_management:investor_imitator",
                "portfolio_management:pg",
                "portfolio_management:ppo",
                "portfolio_management:sac",
                "portfolio_management:sarl",
                "portfolio_management:td3"
            ],
            "start_date": {
                "algorithmic_trading:BTC": "2015-10-01",
                "algorithmic_trading:FX": "2000-01-01",
                "order_execution:BTC": "2021-04-07",
                "order_execution:PD_BTC": "2013-04-29",
                "portfolio_management:dj30": "2012-01-04",
                "portfolio_management:exchange": "2000-01-27",
            },
            "end_date": {
                "algorithmic_trading:BTC": "2021-07-06",
                "algorithmic_trading:FX": "2019-12-31",
                "order_execution:BTC": "2021-03-19",
                "order_execution:PD_BTC": "2021-07-05",
                "portfolio_management:dj30": "2021-12-31",
                "portfolio_management:exchange": "2019-12-31",
            },

        }

    def parameters(self):
        # copy self.parameters
        parameters = copy.deepcopy(self.parameters)
        #TODO
        # update dynamic parameters
        return parameters

    def evaluation_parameters(self,request):
        # print('evaluation_parameters',request)
        request_json = json.loads(request.get_data(as_text=True))
        # print('evaluation_parameters',request_json)

        session_id = request_json.get("session_id")
            #copy the evaluation_parameters_dict
        evaluation_parameters = copy.deepcopy(self.evaluation_parameters_dict)
        self.sessions = self.load_sessions()
        # print('evaluation_parameters', request_json)
        try:
            if 'custom_data_names' in self.sessions[session_id]:
                custom_dataset_names = self.sessions[session_id]["custom_data_names"]
                for dataset_name in custom_dataset_names:
                    data_name_frontend="custom:"+dataset_name
                    evaluation_parameters['dataset_name'].append(data_name_frontend)

                # add parameters for each dataset_name in custom_dataset_names
                for dataset_name in custom_dataset_names:
                    data_name_frontend = "custom:" + dataset_name
                    evaluation_parameters['start_date'][data_name_frontend] = 'Not adjustable'
                    evaluation_parameters['end_date'][data_name_frontend] = 'Not adjustable'
                    evaluation_parameters['labeling_method'][data_name_frontend] = "quantile"
                    evaluation_parameters['merging_metric'][data_name_frontend] = "DTW_distance"
                    evaluation_parameters['merging_threshold'][data_name_frontend] = "0.03"
                    evaluation_parameters['merging_dynamic_constraint'][data_name_frontend] = 1
                    evaluation_parameters['filter_strength'][data_name_frontend] = "1"
                    evaluation_parameters['max_length_expectation'][data_name_frontend] ="144"
                    evaluation_parameters['min_length_limit'][data_name_frontend] = "24"
                    evaluation_parameters['key_indicator'][data_name_frontend] = 'key_indicator'
                    evaluation_parameters['timestamp'][data_name_frontend] = "timestamp"
                    evaluation_parameters['tic'][data_name_frontend] = "data"
            # update dynamic parameters
            if 'dynamic_number' in self.sessions[session_id]:
                dynamic_number = self.sessions[session_id]['dynamic_number']
                evaluation_parameters['test_dynamic_number'] = [str(i) for i in range(dynamic_number)]
        except:
            pass
        # except Exception as e:
        #     error_code = 1
        #     exc_type, exc_obj, exc_tb = sys.exc_info()
        #     fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        #     info = "request data error, {}".format(e)
        #     print(info)
        #     res = {
        #         "error_code": error_code,
        #         "info": info + str(exc_type) + str(fname) + str(exc_tb.tb_lineno),
        #         "session_id": ""
        #     }
        #     return jsonify(res)
        return evaluation_parameters

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
        elif task_name == 'market_dynamics_modeling':
            return os.path.join(ROOT, "tools", "market_dynamics_labeling", "run.py")

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
        param = self.parameters
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
            try:
                start_date = request_json.get("start_date")
            except:
                start_date = None
            try:
                end_date = request_json.get("end_date")
            except:
                end_date = None
            ##TODO
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

            if not start_date or not end_date or (task_name == "algorithmic_trading" and dataset_name == "BTC"):
                train_data = pd.read_csv(os.path.join(ROOT, cfg.data.data_path, "train.csv"), index_col=0)
                val_data = pd.read_csv(os.path.join(ROOT, cfg.data.data_path, "valid.csv"), index_col=0)
                test_data = pd.read_csv(os.path.join(ROOT, cfg.data.data_path, "test.csv"), index_col=0)

            else :
                data = pd.read_csv(os.path.join(ROOT, cfg.data.data_path, "data.csv"), index_col=0)
                data = data[(data["date"] >= start_date) & (data["date"] < end_date)]

                # indexs = range(len(data.index.unique()))
                indexs = data.index.unique()

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
                "dataset":request_json.get("dataset_name"),
                "task_name": task_name,
                "work_dir": work_dir,
                "cfg_path": cfg_path,
                "script_path": train_script_path,
                "train_log_path": log_path,
                "test_log_path": os.path.join(os.path.dirname(log_path), "test_log.txt")
            }})

            cmd = "conda activate TradeMaster && nohup python -u {} --config {} --task_name train --verbose 0 > {} 2>&1 &".format(
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
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            info = "request data error, {}".format(e)
            print(info)
            res = {
                "error_code": error_code,
                "info": info+str(exc_type) + str(fname) + str(exc_tb.tb_lineno),
                "session_id": ""
            }
            return jsonify(res)

    def train_status(self, request):
        request_json = json.loads(request.get_data(as_text=True))
        try:

            self.sessions = self.load_sessions()
            session_id = request_json.get("session_id")
            work_dir = self.sessions[session_id]["work_dir"]


            if session_id in self.sessions:
                if os.path.exists(self.sessions[session_id]["train_log_path"]):
                    cmd = "tail -n 2000 {}".format(self.sessions[session_id]["train_log_path"])
                    info = run_cmd(cmd)
                else:
                    info = ""
            else:
                info = "waitinig for train start"
            if 'train end' in info:
                train_end = True
                with open(osp.join(work_dir, f"Visualization_valid.png"), "rb") as image_file:
                    encoded_string = base64.b64encode(image_file.read())
                    result_plot = str(encoded_string, 'utf-8')
            else:
                train_end = False
                result_plot = None

            # parse every line of info with 'm ' and get the latter part
            info = [line.split('m ')[-1] if 'm ' in line else line for line in info.split('\n')]
            # remove 'nohup: ignoring input' in info
            info = [line for line in info if 'nohup: ignoring input' not in line]
            info = '\n'.join(info)


            res = {
                "info": info,
                "train_end": train_end,
                'result_plot': result_plot
            }
            logger.info("get train status success")
            return jsonify(res)

        except Exception as e:
            error_code = 1
            info = "request data error, {}".format(e)
            res = {
                "error_code": error_code,
                "info": '',
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

            cmd = "conda activate TradeMaster && nohup python -u {} --config {} --task_name test --verbose 0 > {} 2>&1 &".format(
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
            work_dir = self.sessions[session_id]["work_dir"]
            if session_id in self.sessions:
                cmd = "tail -n 2000 {}".format(self.sessions[session_id]["test_log_path"])
                info = run_cmd(cmd)
            else:
                info = 'waiting for test start'

            if 'test end' in info:
                test_end = True
                with open(osp.join(work_dir, f"Visualization_test.png"), "rb") as image_file:
                    encoded_string = base64.b64encode(image_file.read())
                result_plot = str(encoded_string, 'utf-8')
            else:
                test_end = False
                result_plot = None

            # parse every line of info with 'm ' and get the latter part if 'm ' in line else keep the line
            info = [line.split('m ')[-1] if 'm ' in line else line for line in info.split('\n')]
            info = [line for line in info if 'nohup: ignoring input' not in line]
            info = '\n'.join(info)

            res = {
                "info": info,
                "test_end": test_end,
                'result_plot': result_plot
            }
            logger.info("get train status success")
            return jsonify(res)

        except Exception as e:
            error_code = 1
            info = "request data error, {}".format(e)
            res = {
                "error_code": error_code,
                "info": '',
                "session_id": ""
            }
            logger.info(info)
            return jsonify(res)

    def start_market_dynamics_modeling(self, request):
        # there are three ways to start start_market_dynamics_modeling
        # 1. start after the training of agents
        # 2. start from here using customized dataset
        # 3. start from here using the dataset from the training of agents



        request_json = json.loads(request.get_data(as_text=True))
        # print('start_market_dynamics_modeling',request_json)
        try:
            # get input args

            info=''
            # market_dynamics_labeling parameters
            args = {}
            args['dataset_name'] = request_json.get("dataset_name")
            args['number_of_market_style'] = request_json.get("number_of_market_style")
            args['min_length_limit'] = request_json.get("min_length_limit")
            args['filter_strength'] = request_json.get("filter_strength")
            args['max_length_expectation'] = request_json.get("max_length_expectation")
            args['key_indicator'] = request_json.get("key_indicator")
            args['timestamp'] = request_json.get("timestamp")
            args['tic'] = request_json.get("tic")
            args['labeling_method'] = request_json.get("labeling_method")
            args['merging_metric'] = request_json.get("merging_metric")
            args['merging_threshold'] = request_json.get("merging_threshold")
            args['merging_dynamic_constraint'] = request_json.get("merging_dynamic_constraint")

            session_id = request_json.get("session_id")
            # print('request_json',request_json)
            # print('session_id',session_id)
            if session_id is None or session_id == '':
                # if no session_id, create a new session with blank_training
                session_id = self.blank_training(task_name='market_dynamics_modeling',dataset_name=args['dataset_name'])
            # load session
            self.sessions = self.load_sessions()
            if session_id in self.sessions:
                work_dir = self.sessions[session_id]["work_dir"]
                cfg_path = self.sessions[session_id]["cfg_path"]
                print(self.sessions[session_id])
                try:
                    custom_datafile_paths = self.sessions[session_id]["custom_datafile_paths"]
                    custom_datafile_names = ['custom:'+str(os.path.basename(path).split('.')[0]) for path in custom_datafile_paths]
                except:
                    custom_datafile_names = []
                    custom_datafile_paths = []
            # prepare data

            cfg = Config.fromfile(cfg_path)
            cfg = replace_cfg_vals(cfg)

            test_start_date = request_json.get("startDate")
            test_end_date = request_json.get("endDate")
            print(test_start_date, test_end_date)
            # if dataset is custom, change the cfg.data.data_path to custom data path
            # we do net let time slice in custom dataset
            print(custom_datafile_paths,args['dataset_name'], custom_datafile_names)
            if args['dataset_name'] in custom_datafile_names:
                cfg.data.data_path = custom_datafile_paths[custom_datafile_names.index(args['dataset_name'])]
                data=pd.read_csv(os.path.join(ROOT, cfg.data.data_path), index_col=0)
            else:
                # print('cfg.data.data_path',cfg.data.data_path)
                # parse the args['dataset_name'] by : into task_name and dataset_name
                task_name, dataset_name = args['dataset_name'].split(':')
                logger.info('run MDM file_path', os.path.join(ROOT, 'data', task_name, dataset_name, "data.csv"))
                logger.info('timestamp',args['timestamp'])
                try:
                    data = pd.read_csv(os.path.join(ROOT, 'data',task_name,dataset_name, "data.csv"), index_col=0)
                except:
                    data = pd.read_feather(os.path.join(ROOT, 'data',task_name,dataset_name, "data.feather"))
                    # data = data[(data[args['timestamp']] >= test_start_date) & (data[args['timestamp']] < test_end_date)]
                try:
                    data = data[
                        (data[args['timestamp']] >= test_start_date) & (data[args['timestamp']] < test_end_date)]
                except:
                    logger.info('slice data error, please check the timestamp')


            info += f"There are total {data.shape[0]} ticks in the dataset\\n"
            data_path = os.path.join(work_dir, "dynamics_test.csv").replace("\\", "/")
            data.to_csv(data_path)
            args['dataset_path'] = data_path

            # prepare PM index data if needed
            if request_json.get("dataset_name") == 'portfolio_management:dj30':
                DJI_data = pd.read_csv(os.path.join(ROOT, cfg.data.data_path, "DJI.csv"), index_col=0)
                DJI_data = DJI_data[(DJI_data["date"] >= test_start_date) & (DJI_data["date"] < test_end_date)]
                DJI_data_path = os.path.join(work_dir, "DJI_index_dynamics_test.csv").replace("\\", "/")
                DJI_data.to_csv(DJI_data_path)
                args['PM'] = args['dataset_path']
                args['dataset_path']=DJI_data_path
            else:
                args['PM'] = ''


            # update MDM cfg

            MDM_cfg_path = os.path.join(ROOT, "configs", 'market_dynamics_modeling',
                                        f"market_dynamics_modeling.py")

            MDM_cfg = Config.fromfile(MDM_cfg_path)
            MDM_cfg = replace_cfg_vals(MDM_cfg)
            # front-end args to back-end args
            args = MRL_F2B_args_converter(args)
            print(type(args['number_of_market_style']))
            print(args['number_of_market_style'])
            MDM_cfg.market_dynamics_model.update({'data_path': args['dataset_path'],
                                                   'dynamic_number': int(args['number_of_market_style']),
                                                    'min_length_limit': int(args['min_length_limit']),
                                                    'filter_strength': float(args['filter_strength']),
                                                    'max_length_expectation': int(args['max_length_expectation']),
                                                    'key_indicator': args['key_indicator'],
                                                    'timestamp': args['timestamp'],
                                                    'tic': args['tic'],
                                                    'labeling_method': args['labeling_method'],
                                                    'merging_metric': args['merging_metric'],
                                                    'merging_threshold': float(args['merging_threshold']),
                                                    'merging_dynamic_constraint': int(args['merging_dynamic_constraint']),
                                                    'PM': args['PM'],
                                                    'dataset_name': args['dataset_name'],
                                                    'process_datafile_path' : '',
                                                    "market_dynamic_modeling_visualization_paths" : [],
                                                    "market_dynamic_modeling_analysis_paths" : [],
                                                    "exp_name":'default'
                                                  })
            MDM_cfg_path = os.path.join(work_dir, osp.basename(MDM_cfg_path))
            MDM_cfg.dump(MDM_cfg_path)
            logger.info(MDM_cfg)
            MDM_log_path = os.path.join(work_dir, "MDM_log.txt")
            MDM_script_path = self.train_scripts('market_dynamics_modeling', '', '', '', '')
            MDM_dataset_name = args['dataset_name']

            # update session file

            # first test
            if "MDM_cfg_path" not in self.sessions[session_id]:
                self.sessions = self.dump_sessions({session_id: self.sessions[session_id] | {
                    "MDM_cfg_path": MDM_cfg_path,
                    "MDM_script_path": MDM_script_path,
                    "MDM_log_path": MDM_log_path,
                    "MDM_dataset_name": MDM_dataset_name
                }})
            # following dump
            else:
                self.sessions[session_id]["MDM_cfg_path"] = MDM_cfg_path
                self.sessions[session_id]["MDM_script_path"] = MDM_script_path
                self.sessions[session_id]["MDM_log_path"] = MDM_log_path
                self.sessions[session_id]["MDM_dataset_name"] = MDM_dataset_name
                self.sessions = self.dump_sessions({session_id: self.sessions[session_id]})

            # run MDM
            #clear the log file if exists
            if os.path.exists(MDM_log_path):
                os.remove(MDM_log_path)


            cmd = "conda activate TradeMaster && nohup python -u {} --config {} --verbose 0 > {} 2>&1 &".format(
                MDM_script_path,
                MDM_cfg_path,
                MDM_log_path)


            executor.submit(run_cmd, cmd)
            logger.info(cmd)

            # logger.info(cmd)
            # MDM_run_info = run_cmd(cmd)
            # logger.info(MDM_run_info)


            # reload MDM_cfg to get results
            # MDM_cfg = Config.fromfile(MDM_cfg_path)
            # MDM_cfg = replace_cfg_vals(MDM_cfg)
            # MDM_datafile_path = MDM_cfg.market_dynamics_model.process_datafile_path
            # MDM_visualization_paths = MDM_cfg.market_dynamics_model.market_dynamic_modeling_visualization_paths
            # MDM_analysis_path = MDM_cfg.market_dynamics_model.market_dynamic_modeling_analysis_paths
            #
            # with open(MDM_visualization_paths[0], "rb") as image_file:
            #     encoded_string_result = base64.b64encode(image_file.read())
            #
            # with open(MDM_analysis_path[0], "rb") as image_file:
            #     encoded_string_analysis = base64.b64encode(image_file.read())

            # update session information:
            # if "MDM_datafile_path" not in self.sessions[session_id]:
            self.sessions[session_id]["MDM_cfg_path"] = MDM_cfg_path
            self.sessions[session_id]["MDM_script_path"] = MDM_script_path
            self.sessions[session_id]["MDM_log_path"] = MDM_log_path
            self.sessions[session_id]["MDM_dataset_name"] = MDM_dataset_name
            self.sessions = self.dump_sessions({session_id: self.sessions[session_id] })
                # self.sessions = self.dump_sessions({session_id: self.sessions[session_id] |
                #                                                 {
                #                                                     "MDM_datafile_path": MDM_datafile_path,
                #                                                     "MDM_visualization_paths": MDM_visualization_paths}
                #                                     })
            # else:
            #     self.sessions[session_id]["MDM_cfg_path"] = MDM_cfg_path
            #     self.sessions[session_id]["MDM_script_path"] = MDM_script_path
            #     self.sessions[session_id]["MDM_log_path"] = MDM_log_path
            #     # self.sessions[session_id]["MDM_datafile_path"].extend(MDM_datafile_path)
            #     # self.sessions[session_id]["MDM_visualization_paths"].extend(MDM_visualization_paths)
            #     self.sessions[session_id]["MDM_dataset_name"].append(MDM_dataset_name)
            #     self.sessions = self.dump_sessions({session_id: self.sessions[session_id]})

            error_code = 0
            info = "request success, start market dynamics modeling\n"

            res = {
                "error_code": error_code,
                "info": info,
                "session_id": session_id
            }
            logger.info(info)
            return jsonify(res)

        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            error_code = 1
            info = "request data error, {}".format(e)
            res = {
                "error_code": error_code,
                "info": info + str(exc_type) + str(fname) + str(exc_tb.tb_lineno),
                "session_id": session_id
            }
            logger.info(info)
            return jsonify(res)

    def dynamics_testing_status(self, request):
        request_json = json.loads(request.get_data(as_text=True))
        print('dynamics_testing_status', request_json)
        self.sessions = self.load_sessions()
        dynamics_test_label = request_json.get("dynamics_test_label")
        session_id = request_json.get("session_id")
        task_name = self.sessions[session_id]["task_name"]
        try:
            # print('if session_id in self.sessions', session_id in self.sessions)
            # print('MDM_status sessions_1', self.sessions[session_id])
            work_dir = self.sessions[session_id]["work_dir"]
            dt_log_path = os.path.join(work_dir, "dynamics_test_" + str(dynamics_test_label) + "_log.txt")
            if session_id in self.sessions:
                if os.path.exists(dt_log_path):
                    cmd = "tail -n 2000 {}".format(dt_log_path)
                    info = run_cmd(cmd)
                else:
                    info = ""
            else:
                info = "waiting for dynamics testing start"
            if 'dynamics test end' in info:
                dynamics_testing_end = True
                # with open(osp.join(work_dir, f"Visualization_valid.png"), "rb") as image_file:
                #     encoded_string = base64.b64encode(image_file.read())
                #     result_plot = str(encoded_string, 'utf-8')
            else:
                dynamics_testing_end = False
                result_plot = None

            # parse every line of info with 'm ' and get the latter part
            info = [line.split('m ')[-1] if 'm ' in line else line for line in info.split('\n')]
            # remove 'nohup: ignoring input' in info
            info = [line for line in info if 'nohup: ignoring input' not in line]
            info = '\n'.join(info)

            radar_plot_path = osp.join(work_dir, 'radar_plot_agent_' + str(dynamics_test_label) + '.png')
            addtional_info = ""
            if dynamics_testing_end:
                if task_name == "order_execution":
                    radar_plot = None
                    addtional_info += '\n we do not provide radar report for order execution task for now'
                else:
                    with open(radar_plot_path, "rb") as image_file:
                        radar_plot = base64.b64encode(image_file.read())
                        radar_plot= str(radar_plot, 'utf-8')

            else:
                radar_plot = None
            res = {
                'error_code': 0,
                "info": info+addtional_info,
                "dynamics_testing_end": dynamics_testing_end,
                'radar_plot': radar_plot,
                'session_id': session_id
            }
            logger.info("get market dynamics modeling running status success")
            return jsonify(res)

        except Exception as e:
            error_code = 1
            exc_type, exc_obj, exc_tb = sys.exc_info()
            info = "request data error, {}".format(e) + str(exc_type) + str(exc_tb.tb_lineno)
            res = {
                'error_code':1,
                "info": info,
                "dynamics_testing_end": dynamics_testing_end,
                'radar_plot': radar_plot,
                'session_id': session_id
            }
            logger.info(info)
            return jsonify(res)

    def MDM_status(self, request):
        request_json = json.loads(request.get_data(as_text=True))
        self.sessions = self.load_sessions()
        session_id = request_json.get("session_id")
        # print('MDM_status', request_json)
        try:
            # print('if session_id in self.sessions', session_id in self.sessions)
            # print('MDM_status sessions_1', self.sessions[session_id])
            MDM_cfg_path = self.sessions[session_id]["MDM_cfg_path"]
            if session_id in self.sessions:
                if os.path.exists(self.sessions[session_id]["MDM_log_path"]):
                    cmd = "tail -n 2000 {}".format(self.sessions[session_id]["MDM_log_path"])
                    info = run_cmd(cmd)
                else:
                    info = ""
            else:
                info = "waiting for market dynamics modeling start"
            if 'market dynamics modeling analysis done' in info:
                MDM_end = True
                # with open(osp.join(work_dir, f"Visualization_valid.png"), "rb") as image_file:
                #     encoded_string = base64.b64encode(image_file.read())
                #     result_plot = str(encoded_string, 'utf-8')
            else:
                MDM_end = False
                result_plot = None

            # parse every line of info with 'm ' and get the latter part
            info = [line.split('m ')[-1] if 'm ' in line else line for line in info.split('\n')]
            # remove 'nohup: ignoring input' in info
            info = [line for line in info if 'nohup: ignoring input' not in line]
            # if 'it/s]' in line of info, seperate the line with 'it/s]' and get the last segment finished with 'it/s]'
            info = [line.split('it/s]')[-2] if 'it/s]' in line else line for line in info]
            info = '\n'.join(info)




            # reload MDM_cfg to get results
            try:
                #update MDM_cfg if MDM_end
                if MDM_end:
                    MDM_cfg = Config.fromfile(MDM_cfg_path)
                    MDM_cfg = replace_cfg_vals(MDM_cfg)
                    MDM_datafile_path = MDM_cfg.market_dynamics_model.process_datafile_path
                    MDM_visualization_paths = MDM_cfg.market_dynamics_model.market_dynamic_modeling_visualization_paths
                    MDM_analysis_path = MDM_cfg.market_dynamics_model.market_dynamic_modeling_analysis_paths

                    with open(MDM_visualization_paths[0], "rb") as image_file:
                        encoded_string_result = base64.b64encode(image_file.read())
                        encoded_string_result= str(encoded_string_result, 'utf-8')

                    with open(MDM_analysis_path[0], "rb") as image_file:
                        encoded_string_analysis = base64.b64encode(image_file.read())
                        encoded_string_analysis = str(encoded_string_analysis, 'utf-8')

                    # update session information:
                    if "MDM_datafile_path" not in self.sessions[session_id]:
                        self.sessions = self.dump_sessions({session_id: self.sessions[session_id] |
                                                                        {
                                                                            "MDM_datafile_path": [MDM_datafile_path],
                                                                            "MDM_visualization_paths": [path for path in MDM_visualization_paths]}
                                                            })
                    else:
                        self.sessions[session_id]["MDM_datafile_path"].append(MDM_datafile_path)
                        self.sessions[session_id]["MDM_visualization_paths"].extend(MDM_visualization_paths)
                        self.sessions = self.dump_sessions({session_id: self.sessions[session_id]})
                else:
                    encoded_string_result = None
                    encoded_string_analysis = None
            except:
                encoded_string_result = None
                encoded_string_analysis = None




            res = {
                "info": info,
                "MDM_end": MDM_end,
                "market_dynamic_labeling_visulization": encoded_string_result,
                "market_dynamic_labeling_analysis": encoded_string_analysis,
                'session_id': session_id
            }
            logger.info("get market dynamics modeling running status success")
            return jsonify(res)

        except Exception as e:
            error_code = 1
            exc_type, exc_obj, exc_tb = sys.exc_info()
            info = "request data error, {}".format(e) + str(exc_type) + str(exc_tb.tb_lineno)
            res = {
                "error_code": error_code,
                "info": info,
                "market_dynamic_labeling_visulization": None,
                "market_dynamic_labeling_analysis": None,
                "session_id": session_id
            }
            logger.info(info)
            return jsonify(res)



    def save_market_dynamics_labeling(self, request):
        request_json = json.loads(request.get_data(as_text=True))
        try:
            session_id = request_json.get("session_id")

            self.sessions = self.load_sessions()
            if session_id in self.sessions:
                work_dir = self.sessions[session_id]["work_dir"]
                cfg_path = self.sessions[session_id]["cfg_path"]
                MDM_datafile_path = self.sessions[session_id]["MDM_datafile_path"]
                MDM_dataset_name = self.sessions[session_id]["MDM_dataset_name"]
                MDM_cfg_path=self.sessions[session_id]["MDM_cfg_path"]

            cfg = Config.fromfile(cfg_path)
            cfg = replace_cfg_vals(cfg)


            # get the dynamic_number from cfg read from MDM_cfg_path
            MDM_cfg = Config.fromfile(MDM_cfg_path)
            MDM_cfg = replace_cfg_vals(MDM_cfg)
            dynamic_number = MDM_cfg.market_dynamics_model.dynamic_number
            # store the dynamic_number in session information
            self.sessions[session_id]["dynamic_number"] = dynamic_number
            self.sessions = self.dump_sessions({session_id: self.sessions[session_id]})

            # print('save market dynamics labeling',self.sessions[session_id])
            # logger.info('save market dynamics labeling',self.sessions[session_id])

            # print('MDM_datafile_path',MDM_datafile_path)
            # logger.info('MDM_datafile_path',MDM_datafile_path)
            # update dataset cfg
            cfg.data.test_dynamic_path = MDM_datafile_path[-1].replace("\\", "/")
            # print('cfg.data.test_dynamic_path',cfg.data.test_dynamic_path)
            # logger.info('cfg.data.test_dynamic_path',cfg.data.test_dynamic_path)
            cfg_path = os.path.join(work_dir, osp.basename(cfg_path))
            cfg.dump(cfg_path)
            logger.info(cfg)
            error_code = 0
            info = "request success, save market dynamics"
            res = {
                "error_code": error_code,
                "info": info,
                'session_id':session_id,
                'debug': self.sessions[session_id]
            }
            logger.info(info)
            return jsonify(res)

        except Exception as e:
            error_code = 1
            info = "request data error, {}".format(e)
            res = {
                "error_code": error_code,
                "info": info,
                'session_id':session_id
            }
            logger.info(info)
            return jsonify(res)

    def run_dynamics_test(self, request):
        request_json = json.loads(request.get_data(as_text=True))
        # print('run dynamics test',request_json)
        try:
            dynamics_test_label = request_json.get("dynamics_test_label")
            session_id = request_json.get("session_id")
            logger.info(request_json)

            self.sessions = self.load_sessions()
            addtional_info = ''
            if session_id in self.sessions:
                work_dir = self.sessions[session_id]["work_dir"]
                cfg_path = self.sessions[session_id]["cfg_path"]
                train_script_path = self.sessions[session_id]["script_path"]
                task_name = self.sessions[session_id]["task_name"]
            # no agent to be tested if task_name is market_dynamics_labeling

            if task_name == "market_dynamics_labeling":
                # throw error
                error_code = 1
                info = "Illegal instruction, {}".format("no agent to be tested, you may start the from the training page to train an agent")
                res = {
                    "error_code": error_code,
                    "info": info
                }
                logger.info(info)
                return jsonify(res)

            cfg = Config.fromfile(cfg_path)
            cfg = replace_cfg_vals(cfg)
            cfg_path = os.path.join(work_dir, osp.basename(cfg_path))
            dt_log_path = os.path.join(work_dir, "dynamics_test_" + str(dynamics_test_label) + "_log.txt")
            cmd = "conda activate TradeMaster && nohup python -u {} --config {} --task_name dynamics_test --test_dynamic {} --verbose 0 > {} 2>&1 &".format(
                train_script_path,
                cfg_path,
                dynamics_test_label,
                dt_log_path)
            logger.info(cmd)

            executor.submit(run_cmd, cmd)
            logger.info(cmd)


            # DT_info = run_cmd(cmd)
            # logger.info(DT_info)
            # radar_plot_path = osp.join(work_dir, 'radar_plot_agent_' + str(dynamics_test_label) + '.png')
            # if task_name == "order_execution":
            #     encoded_string = ""
            #     addtional_info += '\n we do not provide radar report for order execution task for now'
            # else:
            #     with open(radar_plot_path, "rb") as image_file:
            #         encoded_string = base64.b64encode(image_file.read())

            # # print log output
            # print_log_cmd = "tail -n 100 {}".format(dt_log_path)
            # dynamics_test_log_info = run_cmd(print_log_cmd)

            error_code = 0
            info = f"request success, start test market {dynamics_test_label}\n\n"
            res = {
                "error_code": error_code,
                # "info": info + dynamics_test_log_info + addtional_info,
                "info": info,
                "session_id": session_id,
                # 'radar_plot': str(encoded_string, 'utf-8')
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
                # 'radar_plot': ""
            }
            logger.info(info)
            return jsonify(res)


    def upload_csv(self, request):
        request_json = json.loads(request.get_data(as_text=True))
        try:
            file_name=request_json.get('file_name')
            custom_data_name = file_name.split('.')[0]
            custom_data=request_json.get('file')
            if file_name == '' or file_name is None:
                return jsonify({'error': 'No file selected'}), 400
            if file_name and file_name.endswith('.csv') is False:
                return jsonify({'error': 'Only support csv file'}), 400



            session_id = request_json.get("session_id")
            # print('get session_id from request_json', session_id)
            if session_id is None or session_id=='':
                session_id = self.blank_training(
                    task_name="market_dynamics_modeling",
                    dataset_name="custom",
                    optimizer_name='adam',
                    loss_name="mse",
                    agent_name="Null")
                print('create new session_id', session_id)


            self.sessions = self.load_sessions()
            if session_id in self.sessions:
                work_dir = self.sessions[session_id]["work_dir"]
            # save custom data to csv file in work_dir
            custom_datafile_path = os.path.join(work_dir, custom_data_name + ".csv")
            print('custom_datafile_path', custom_datafile_path)

            # Parse the string into a list of dictionaries
            data = json.loads(custom_data)

            # Write the data to the CSV file with replace the old one

            with open(custom_datafile_path, 'w',newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=data[0].keys())
                writer.writeheader()
                for row in data:
                    writer.writerow(row)


            # update session info
            if "custom_datafile_paths" not in self.sessions[session_id]:
                self.sessions = self.dump_sessions({session_id: self.sessions[session_id] |
                                                                {
                                                                    "custom_datafile_paths":[custom_datafile_path],
                                                                'custom_data_names': [custom_data_name]
                                                                }
                                                    })
            else:
                if custom_datafile_path not in self.sessions[session_id]["custom_datafile_paths"]:
                    self.sessions[session_id]["custom_datafile_paths"].append(custom_datafile_path)
                    self.sessions[session_id]["custom_data_names"].append(custom_data_name)
                    self.sessions = self.dump_sessions({session_id: self.sessions[session_id]})


            error_code = 0
            info = "request success, read uploaded csv file"
            res = {
                "error_code": error_code,
                "info": info,
                "session_id": session_id,
                'upload_finished': True

            }
            logger.info(info)





            return jsonify(res)

        except Exception as e:
            error_code = 1
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            info = "request data error, {}".format(e) + str(exc_type) + str(fname) + str(exc_tb.tb_lineno)
            res = {
                "error_code": error_code,
                "info": info,
                "session_id": ""
            }
            logger.info(info)
            return jsonify(res)


    def blank_training(self, task_name ="custom",
                       dataset_name = "custom",
                       optimizer_name = 'adam',
                       loss_name = "mse",
                       agent_name = "Null"
                       ):


        ##TODO
        try:
            session_id = str(uuid.uuid1())

            cfg_path = os.path.join(ROOT, "configs", task_name,
                                    f"{task_name}_any_{agent_name}_{agent_name}_{optimizer_name}_{loss_name}.py")
            train_script_path = self.train_scripts(task_name, dataset_name, optimizer_name, loss_name, agent_name)
            work_dir = os.path.join(ROOT, "work_dir", session_id,
                                    f"{task_name}_any_{agent_name}_{agent_name}_{optimizer_name}_{loss_name}")
            if not os.path.exists(work_dir):
                os.makedirs(work_dir)


            cfg = Config.fromfile(cfg_path)
            cfg = replace_cfg_vals(cfg)
            cfg.work_dir = "work_dir/{}/{}".format(session_id,
                                                   task_name)
            cfg.trainer.work_dir = cfg.work_dir

            #update data path according to dataset_name
            if dataset_name == "custom":
                task_name, tic_name= dataset_name, dataset_name
            else:
                task_name, tic_name = dataset_name.split(':')
            # check the extension name of the data file
            cfg.data.data_path = "data/{}/{}".format(task_name, tic_name)
            cfg.data.train_path = "{}/{}".format(cfg.work_dir, "train.csv")
            # val_data.to_csv(os.path.join(work_dir, "valid.csv"))
            cfg.data.valid_path = "{}/{}".format(cfg.work_dir, "valid.csv")
            # test_data.to_csv(os.path.join(work_dir, "test.csv"))
            cfg.data.test_path = "{}/{}".format(cfg.work_dir, "test.csv")

            cfg_path = os.path.join(work_dir, osp.basename(cfg_path))
            cfg.dump(cfg_path)
            logger.info(cfg)

            log_path = os.path.join(work_dir, "train_log.txt")

            self.sessions = self.dump_sessions({session_id: {
                "dataset": dataset_name,
                "task_name": task_name,
                "work_dir": work_dir,
                "cfg_path": cfg_path,
                "script_path": train_script_path,
                "train_log_path": log_path,
                "test_log_path": os.path.join(os.path.dirname(log_path), "test_log.txt")
            }})

            return session_id
        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            info = str(exc_type) + str(fname) + str(exc_tb.tb_lineno)
            print(info)
            return None


    def download_csv(self, request):
        request_json = json.loads(request.get_data(as_text=True))
        # print('request_json',request_json)
        try:
            session_id = request_json.get("session_id")
            # print('request_json',request_json)
            # print('download_session',self.sessions[session_id])
            self.sessions = self.load_sessions()
            if session_id in self.sessions:
                work_dir = self.sessions[session_id]["work_dir"]
                # get the saved csv file path that is set in save_market_dynamics_labeling()
                # TODO: modify the save_market_dynamics_labeling()
                # get all custom data file paths
                if "custom_datafile_paths" in self.sessions[session_id]:
                    custom_datafile_paths = self.sessions[session_id]["custom_datafile_paths"]
                    # get the file name without extension
                    # parse all custom data file name from custom_datafile_paths
                    custom_datafile_names = [os.path.basename(path).split('.')[0] for path in custom_datafile_paths]
                else:
                    custom_datafile_paths = []
                    custom_datafile_names = []
                MDM_datafile_path = self.sessions[session_id]["MDM_datafile_path"][-1]
                # MDM_dataset_name = self.sessions[session_id]["MDM_dataset_name"]
                MDM_dataset_name = str(os.path.basename(MDM_datafile_path).split('.')[0])
            else:
                error_code = 1
                info = "Please start your session first from running a modeling"
                res = {
                    "error_code": error_code,
                    "info": info,
                    "file": None
                }
                logger.info(info)
                return jsonify(res)



            # forbid download csv file if it is not a custom data
            # if MDM_dataset_name not in custom_datafile_names:
            #     error_code = 1
            #     info = "request error, you can only download custom data"
            #     res = {
            #         "error_code": error_code,
            #         "info": info,
            #         "file": None
            #     }
            #     logger.info(info)
            #     return jsonify(res)

            # get the saved csv file path that is set in save_market_dynamics_labeling()
            # print('saved_dataset_path',self.sessions[session_id]["MDM_datafile_path"])
            saved_dataset_path = self.sessions[session_id]["MDM_datafile_path"][-1]

            # Read the CSV file contents
            with open(saved_dataset_path, 'r', encoding='utf-8') as file:
                csv_data = file.read()

            # Create a response with appropriate headers for downloading a CSV file

            # TODO: modify the file response name
            # response = Response(
            #     csv_data,
            #     content_type='text/csv',
            #     headers={
            #         "Content-Disposition": f"attachment; filename={MDM_dataset_name}",
            #         "Content-Type": "text/csv; charset=utf-8"
            #     }
            # )

            error_code = 0
            info = "request success, read uploaded csv file"
            res = {
                "error_code": error_code,
                "info": info,
                "data": csv_data,
                'filename': MDM_dataset_name,
                'session_id': session_id
            }
            logger.info(info)
            return jsonify(res)

        except Exception as e:

            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            error_code = 1
            info = "request data error, {}".format(e)
            res = {
                "error_code": error_code,
                "info": info + str(exc_type) + str(fname) + str(exc_tb.tb_lineno),
                "session_id": session_id,
                "file": None
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

@app.route("/api/TradeMaster/upload_csv", methods=['POST'])
def upload_csv():
    res = SERVER.upload_csv(request)
    return res

@app.route("/api/TradeMaster/download_csv", methods=['POST'])
def download_csv():
    res = SERVER.download_csv(request)
    return res

@app.route("/api/TradeMaster/getParameters", methods=["GET"])
def getParameters():
    res = SERVER.get_parameters(request)
    return res

@app.route("/api/TradeMaster/evaluation_getParameters", methods=["POST"])
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


@app.route("/api/TradeMaster/dynamics_testing_status", methods=["POST"])
def dynamics_testing_status():
    res = SERVER.dynamics_testing_status(request)
    return res


@app.route("/api/TradeMaster/MDM_status", methods=["POST"])
def MDM_status():
    res = SERVER.MDM_status(request)
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
    res = SERVER.start_market_dynamics_modeling(request)
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
    host = "127.0.0.1"
    port = 8080
    app.run(host, port)
