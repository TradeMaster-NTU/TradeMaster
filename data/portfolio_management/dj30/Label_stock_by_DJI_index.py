import numpy as np
import pandas as pd

DJI_index_path = 'data/portfolio_management/dj30/Market_Dynamics_Model/DJI/DJI_labeled_slice_and_merge_model_3dynamics_minlength12_quantile_labeling.csv'
stock_path= 'data/portfolio_management/dj30/test.csv'

# get the label column from the DJI index and add it to the stock data by merging on the date column

# read the stock data
stock_data = pd.read_csv(stock_path)
# read the DJI index data
DJI_index_data = pd.read_csv(DJI_index_path)
# add the label column to the stock data with the same date
stock_data = pd.merge(stock_data, DJI_index_data[['date', 'label']], on='date', how='left')
# write the stock data with the label column to a csv file
stock_data.to_csv('data/portfolio_management/dj30/test_with_label.csv', index=False)


