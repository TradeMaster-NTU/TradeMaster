import argparse
import json
import os
import sys
import pytz
import requests
import time

tz = pytz.timezone('Asia/Shanghai')
root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(root)

import random
import string


def generate_random_str(randomlength=16):
    str_list = [random.choice(string.digits + string.ascii_letters) for i in range(randomlength)]
    random_str = ''.join(str_list)
    return random_str


def test_train(request_url):
    headers = {
        "Content-Type": "application/json"
    }
    try:
        data = {
            "task_name": "algorithmic_trading",
            "dataset_name": "algorithmic_trading:BTC",
            "train_start_date":"2017-01-01",
            "test_start_date":"2019-12-31",
            "optimizer_name": "adam",
            "loss_name": "mse",
            "agent_name": "algorithmic_trading:dqn",
        }
        data = json.dumps(data)
        response_return = requests.post(request_url, data=data, headers=headers).text
        return response_return
    except Exception as e:
        print(e)

def test_train_status(request_url, session_id):
    headers = {
        "Content-Type": "application/json"
    }
    try:
        data = {
            "session_id":session_id
        }
        data = json.dumps(data)
        response_return = requests.post(request_url, data=data, headers=headers).text
        return response_return
    except Exception as e:
        print(e)

def test_test(request_url, session_id):
    headers = {
        "Content-Type": "application/json"
    }
    try:
        data = {
            "session_id":session_id
        }
        data = json.dumps(data)
        response_return = requests.post(request_url, data=data, headers=headers).text
        return response_return
    except Exception as e:
        print(e)

def test_test_status(request_url, session_id):
    headers = {
        "Content-Type": "application/json"
    }
    try:
        data = {
            "session_id":session_id
        }
        data = json.dumps(data)
        response_return = requests.post(request_url, data=data, headers=headers).text
        return response_return
    except Exception as e:
        print(e)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-ht', '--host', type=str, default="20a6p05819.goho.co", help='request host')
    parser.add_argument('-pt', '--port', type=int, default=36712, help='request port')
    parser.add_argument('-tn', '--turn', type=int, default=1, help='request turn')

    args = parser.parse_args()

    host = str(args.host)
    port = int(args.port)
    turn = int(args.turn)

    url = "http://{}:{}/TradeMaster/train".format(host, port)
    response_return = test_train(url)
    print(response_return)

    time.sleep(10)

    # session_id = response_return["session_id"]
    # url = "http://{}:{}/TradeMaster/train_status".format(host, port)
    # response_return = test_train_status(url, session_id)
    # print(response_return)
    #
    # time.sleep(100)
    #
    # url = "http://{}:{}/TradeMaster/test".format(host, port)
    # response_return = test_test(url, session_id)
    # print(response_return)
    #
    # time.sleep(10)
    #
    # url = "http://{}:{}/TradeMaster/test_status".format(host, port)
    # response_return = test_test_status(url, session_id)
    # print(response_return)


