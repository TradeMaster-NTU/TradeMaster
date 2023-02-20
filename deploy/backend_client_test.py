import os
import sys
import requests
import json
import base64
import pytz
from glob import glob
import argparse

tz = pytz.timezone('Asia/Shanghai')
root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(root)

import random
import string

def generate_random_str(randomlength=16):
    str_list = [random.choice(string.digits + string.ascii_letters) for i in range(randomlength)]
    random_str = ''.join(str_list)
    return random_str

def test_getParameters(request_url):

    headers = {
        "Content-Type":"application/json"
    }
    try:
        response_return = requests.get(request_url,headers = headers).text
        print(response_return)
    except Exception as e:
        print(e)


if __name__=="__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-ht','--host', type= str, default="119.28.134.35", help='request host')
    parser.add_argument('-pt','--port', type= int , default=443, help='request port')
    parser.add_argument('-tn','--turn', type = int, default=1, help='request turn')

    args = parser.parse_args()

    host = str(args.host)
    port = int(args.port)
    turn = int(args.turn)

    url = "http://{}:{}/TradeMaster/getParameters".format(host, port)
    test_getParameters(url)