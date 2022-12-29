import random
import torch
import os
import numpy as np
import yaml
import pandas as pd
import copy

def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def load_yaml(yaml_path):
    curPath = os.path.abspath('.')
    yaml_path = os.path.join(curPath, yaml_path)
    f = open(yaml_path, 'r', encoding='utf-8')
    cfg = f.read()
    d = yaml.load(cfg, Loader=yaml.FullLoader)
    return d

def load_style_yaml(yaml_path,style):
    curPath = os.path.abspath('.')
    yaml_path = os.path.join(curPath, yaml_path)
    f = open(yaml_path, 'r', encoding='utf-8')
    cfg = f.read()
    d = yaml.load(cfg, Loader=yaml.FullLoader)

    data=pd.read_csv(d["df_path"], index_col=0)
    def get_intervals(data):
        index = data.index
        last_value = index[0] - 1
        last_index = 0
        intervals = []
        for i in range(data.shape[0]):
            if last_value != index[i] - 1:
                intervals.append([last_index, i])
                last_value = index[i]
                last_index = i
            last_value = index[i]
        return intervals
    intervals=get_intervals(data)
    os.makedir('temp')
    d_list=[]
    for i,interval in enumerate(intervals):
        data.iloc[interval[0]:interval[1],:].to_csv('temp/'+str(style)+'_'+str(i)+'.csv')
        temp_d=copy.deepcopy(d)
        temp_d["df_path"]='temp/'+str(style)+'_'+str(i)+'.csv'
        d_list.append(temp_d)
    return d_list