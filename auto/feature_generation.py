import pandas as pd
from sklearn.datasets import fetch_california_housing
from openfe import openfe, tree_to_formula, transform
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
import pdb
import sys
import numpy as np
sys.path.append(".")

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--dataset",
                    type=str,
                    default="BTC",
                    help="the name of dataset for feature generation")
args = parser.parse_args()

def get_score(train_x, test_x, train_y, test_y):
    train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size=1/9, random_state=1)
    params = {'n_estimators': 1000, 'n_jobs': 4, 'seed': 1}
    gbm = lgb.LGBMRegressor(**params)
    gbm.fit(train_x, train_y, eval_set=[(val_x, val_y)], callbacks=[lgb.early_stopping(50, verbose=False)])
    pred = pd.DataFrame(gbm.predict(test_x), index=test_x.index)
    score = mean_squared_error(test_y, pred)
    return score

def feature_generation_BTC():
    n_jobs = 4
    train_x = pd.read_csv('data/data/BTC/train.csv')
    train_remain = train_x[['tic','date']]
    train_x = train_x.drop(['Unnamed: 0','tic','date'],axis=1)
    train_y = train_x['adjcp'].diff(periods=-1) * (-1)
    train_y = train_y.fillna(axis=0,method="ffill")

    valid_x = pd.read_csv('data/data/BTC/valid.csv')
    valid_remain = valid_x[['tic','date']]
    valid_x = valid_x.drop(['Unnamed: 0','tic','date'],axis=1)
    valid_y = valid_x['adjcp'].diff(periods=-1) * (-1)
    valid_y = valid_y.fillna(axis=0,method="ffill")

    test_x = pd.read_csv('data/data/BTC/valid.csv')
    test_remain = test_x[['tic','date']]
    test_x = test_x.drop(['Unnamed: 0','tic','date'],axis=1)
    test_y = test_x['adjcp'].diff(periods=-1) * (-1)
    test_y = test_y.fillna(axis=0,method="ffill")

    train_size = train_x.shape[0]
    valid_size = valid_x.shape[0]

    train_x = pd.concat([train_x, valid_x], ignore_index=True)
    train_y = pd.concat([train_y, valid_y], ignore_index=True)
    # get baseline score
    score1 = get_score(train_x, test_x, train_y, test_y)
    # feature generation
    ofe = openfe()
    ofe.fit(data=train_x, label=train_y, n_jobs=n_jobs)


    train_x, test_x = transform(train_x, test_x, ofe.new_features_list[:100], n_jobs=n_jobs)
    score2 = get_score(train_x, test_x, train_y, test_y)
    train = train_x.iloc[0:train_size]
    valid = train_x.iloc[train_size:train_size+valid_size]
    valid = valid.reset_index(drop=True)
    train = pd.concat([train_remain, train], axis=1)
    valid = pd.concat([valid_remain, valid], axis=1)
    test = pd.concat([test_remain, test_x], axis=1)
    train.to_csv('data/data/BTC/train_feature.csv')
    valid.to_csv('data/data/BTC/valid_feature.csv')
    test.to_csv('data/data/BTC/test_feature.csv')
    print("The top generated features are")
    for feature in ofe.new_features_list[:100]:
        print(tree_to_formula(feature))
    print("The MSE before feature generation is", score1)
    print("The MSE after feature generation is", score2)


def feature_generation_dj30():
    n_jobs = 4
    train_x = pd.read_csv('data/data/dj30/train.csv')
    train_remain = train_x[['Unnamed: 0','tic','date']]
    train_x[['label']] = train_x.groupby(['tic']).transform(lambda x: x.diff(periods=-1))[['adjcp']]
    train_x = train_x.groupby(['tic']).fillna(axis=0,method="ffill") 
    train_y = train_x['label'] * (-1)
    train_x = train_x.drop(['Unnamed: 0','date','label'],axis=1)

    valid_x = pd.read_csv('data/data/dj30/valid.csv')
    valid_remain = valid_x[['Unnamed: 0','tic','date']]
    valid_x[['label']] = valid_x.groupby(['tic']).transform(lambda x: x.diff(periods=-1))[['adjcp']]
    valid_x = valid_x.groupby(['tic']).fillna(axis=0,method="ffill") 
    valid_y = valid_x['label'] * (-1)
    valid_x = valid_x.drop(['Unnamed: 0','date','label'],axis=1)

    test_x = pd.read_csv('data/data/dj30/test.csv')
    test_remain = test_x[['Unnamed: 0','tic','date']]
    test_x[['label']] = test_x.groupby(['tic']).transform(lambda x: x.diff(periods=-1))[['adjcp']]
    test_x = test_x.groupby(['tic']).fillna(axis=0,method="ffill") 
    test_y = test_x['label'] * (-1)
    test_x = test_x.drop(['Unnamed: 0','date','label'],axis=1)

    train_size = train_x.shape[0]
    valid_size = valid_x.shape[0]
    train_x = pd.concat([train_x, valid_x], ignore_index=True)
    train_y = pd.concat([train_y, valid_y], ignore_index=True)
    # get baseline score
    score1 = get_score(train_x, test_x, train_y, test_y)
    # feature generation
    ofe = openfe()
    ofe.fit(data=train_x, label=train_y, n_jobs=n_jobs)


    train_x, test_x = transform(train_x, test_x, ofe.new_features_list[:5], n_jobs=n_jobs)
    score2 = get_score(train_x, test_x, train_y, test_y)
    train = train_x.iloc[0:train_size]
    valid = train_x.iloc[train_size:train_size+valid_size]
    valid = valid.reset_index(drop=True)
    train = pd.concat([train_remain, train], axis=1)
    train.set_index('Unnamed: 0',inplace=True)
    valid = pd.concat([valid_remain, valid], axis=1)
    valid.set_index('Unnamed: 0',inplace=True)
    test = pd.concat([test_remain, test_x], axis=1)
    test.set_index('Unnamed: 0',inplace=True)
    train.to_csv('data/data/dj30/train_feature.csv')
    valid.to_csv('data/data/dj30/valid_feature.csv')
    test.to_csv('data/data/dj30/test_feature.csv')
    print("The top generated features are")
    for feature in ofe.new_features_list[:5]:
        print(tree_to_formula(feature))
    print("The MSE before feature generation is", score1)
    print("The MSE after feature generation is", score2)

if __name__ == '__main__':
    if args.dataset == "BTC":
        feature_generation_BTC()
    elif args.dataset == "dj30":
        feature_generation_dj30()