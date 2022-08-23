import random
import torch
import os
import numpy as np
import yaml


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