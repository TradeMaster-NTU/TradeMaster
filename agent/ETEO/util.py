import random
import torch
import os
import numpy as np
import yaml
import copy
import pandas as pd
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
    data=pd.read_csv(d["df_path"]).reset_index()
    data=data.loc[data['label'] == style, :]
    def get_styled_intervals(data):
        index = data['index'].to_list()
        last_value = index[0] - 1
        last_index = 0
        intervals = []
        for i in range(data.shape[0]):
            if last_value != index[i] - 1:
                intervals.append([last_index, i])
                last_value = index[i]
                last_index = i
            last_value = index[i]
        intervals.append([last_index, i])
        return intervals
    intervals=get_styled_intervals(data)
    data.drop(columns=['index'])
    if not os.path.exists('temp'):
        os.makedirs('temp')
    d_list=[]
    for i,interval in enumerate(intervals):
        data.iloc[interval[0]:interval[1],:].to_csv('temp/'+str(style)+'_'+str(i)+'.csv')
        temp_d=copy.deepcopy(d)
        temp_d["df_path"]='temp/'+str(style)+'_'+str(i)+'.csv'
        d_list.append(temp_d)
    return d_list