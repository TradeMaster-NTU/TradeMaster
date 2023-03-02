import pickle
import pdb
import os
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset


attributes = ['open', 'high', 'low', 'close', 'adjcp']


def parse_data(x):
    x = x.set_index("Attribute").to_dict()["Value"]

    values = []

    for attr in attributes:
        if x.__contains__(attr):
            values.append(x[attr])
        else:
            values.append(np.nan)
    return values


def parse_id(missing_ratio=0.1, dataset_name = "dj30", tic_name="AAPL"):
    #dataset_path = "./source/" + dataset_name + ".csv"
    if dataset_name == "dj30":
        dataset_path = "./data/portfolio_management/dj30/dj30.csv"
    elif dataset_name == "exchange":
        dataset_path = "./data/portfolio_management/exchange/exchange.csv"
    elif dataset_name == "btc":
        dataset_path = "./data/algorithmic_trading/BTC/BTC.csv"
    data = pd.read_csv(dataset_path)
    date_list = data.date.unique().tolist()
    start_date = date_list[0]
    end_date = date_list[-1]
    date_list_2 = pd.bdate_range(start=start_date, end=end_date)
    df = data.loc[data['tic']==tic_name]
    df = df[['date', 'open', 'high', 'low', 'close', 'adjcp']]
    df = pd.melt(df, id_vars=["date"], value_vars=['date', 'open', 'high', 'low', 'close', 'adjcp'], var_name="Attribute", value_name="Value")
    df = df.sort_values(by=["date"])
    date = [ x.strftime('%F') for x in date_list_2]
    observed_values = []
    for d in date:
        observed_values.append(parse_data(df[df["date"] == d]))
    observed_values = np.array(observed_values)
    observed_masks = ~np.isnan(observed_values)
    masks = observed_masks.reshape(-1).copy()
    obs_indices = np.where(masks)[0].tolist()
    miss_indices = np.random.choice(
        obs_indices, (int)(len(obs_indices) * missing_ratio), replace=False
    )
    #pdb.set_trace()
    masks[miss_indices] = False
    gt_masks = masks.reshape(observed_masks.shape)

    observed_values = np.nan_to_num(observed_values)
    observed_masks = observed_masks.astype("float32")
    gt_masks = gt_masks.astype("float32")
    return observed_values, observed_masks, gt_masks




class Own_Dataset(Dataset):
    def __init__(self, eval_length=10, use_index_list=None, missing_ratio=0.0, seed=0, dataset_name = "dj30", tic_name = "AAPL"):
        self.eval_length = eval_length
        np.random.seed(seed)  
        self.observed_values = []
        self.observed_masks = []
        self.gt_masks = []
        path = (
            "./work_dir/missing_value_imputation/data/" + dataset_name + "/" + tic_name + "_missing" + str(missing_ratio) + "_seed" + str(seed) + ".pk"
        )

        if os.path.isfile(path) == False:  

            observed_values, observed_masks, gt_masks = parse_id(
                        missing_ratio, dataset_name, tic_name
                )
            self.observed_values.append(observed_values)
            self.observed_masks.append(observed_masks)
            self.gt_masks.append(gt_masks)

            self.observed_values = np.array(self.observed_values)
            self.observed_masks = np.array(self.observed_masks)
            self.gt_masks = np.array(self.gt_masks)

            tmp_values = self.observed_values.reshape(-1, 5)
            tmp_masks = self.observed_masks.reshape(-1, 5)
            mean = np.zeros(5)
            std = np.zeros(5)
            for k in range(5):
                c_data = tmp_values[:, k][tmp_masks[:, k] == 1]
                mean[k] = c_data.mean()
                std[k] = c_data.std()
            
            self.observed_values = (
                (self.observed_values - mean) / std * self.observed_masks
            )

            with open(path, "wb") as f:
                pickle.dump(
                    [self.observed_values, self.observed_masks, self.gt_masks, mean, std], f
                )
            with open(path, "rb") as f:
                self.observed_values, self.observed_masks, self.gt_masks, mean, std= pickle.load(
                    f
                )
            data_num = self.observed_values[0].shape[0]
            data_deleted = data_num % 10
            
            self.observed_values = np.delete(self.observed_values[0], np.s_[-data_deleted: ], axis = 0)
            self.observed_values = np.reshape(self.observed_values, (data_num // 10, 10, 5))
            self.observed_masks = np.delete(self.observed_masks[0], np.s_[-data_deleted: ], axis = 0)
            self.observed_masks = np.reshape(self.observed_masks, (data_num // 10, 10, 5))
            self.gt_masks = np.delete(self.gt_masks[0], np.s_[-data_deleted: ], axis = 0)
            self.gt_masks = np.reshape(self.gt_masks, (data_num // 10, 10, 5))
        else:  
            with open(path, "rb") as f:
                self.observed_values, self.observed_masks, self.gt_masks, mean, std= pickle.load(
                    f
                )
            data_num = self.observed_values[0].shape[0]
            data_deleted = data_num % 10
            self.observed_values = np.delete(self.observed_values[0], np.s_[-data_deleted: ], axis = 0)
            self.observed_values = np.reshape(self.observed_values, (data_num // 10, 10, 5))
            self.observed_masks = np.delete(self.observed_masks[0], np.s_[-data_deleted: ], axis = 0)
            self.observed_masks = np.reshape(self.observed_masks, (data_num // 10, 10, 5))
            self.gt_masks = np.delete(self.gt_masks[0], np.s_[-data_deleted: ], axis = 0)
            self.gt_masks = np.reshape(self.gt_masks, (data_num // 10, 10, 5))
        if use_index_list is None:
            self.use_index_list = np.arange(len(self.observed_values))
        else:
            self.use_index_list = use_index_list

    def __getitem__(self, org_index):
        index = self.use_index_list[org_index]
        s = {
            "observed_data": self.observed_values[index],
            "observed_mask": self.observed_masks[index],
            "gt_mask": self.gt_masks[index],
            "timepoints": np.arange(self.eval_length),
        }
        return s

    def __len__(self):
        return len(self.use_index_list)


def get_dataloader(seed=1, batch_size=16, missing_ratio=0.1, dataset_name = "dj30", tic_name = "AAPL"):


    dataset = Own_Dataset(missing_ratio=missing_ratio, seed=seed, dataset_name = dataset_name, tic_name = tic_name)
    indlist = np.arange(len(dataset))

    np.random.seed(seed)

    start = 0
    end = len(dataset)
    test_index = indlist[start:end]
    remain_index = indlist
    np.random.seed(seed)
    num_train = (int)(len(dataset) * 0.8)
    train_index = remain_index[:num_train]
    valid_index = remain_index[num_train:]

    dataset = Own_Dataset(
        use_index_list=train_index, missing_ratio=missing_ratio, seed=seed, dataset_name = dataset_name, tic_name = tic_name
    )
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=1)
    valid_dataset = Own_Dataset(
        use_index_list=valid_index, missing_ratio=missing_ratio, seed=seed, dataset_name = dataset_name, tic_name = tic_name
    )
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=0)
    test_dataset = Own_Dataset(
        use_index_list=test_index, missing_ratio=missing_ratio, seed=seed, dataset_name = dataset_name, tic_name = tic_name
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=0)
    return train_loader, valid_loader, test_loader
