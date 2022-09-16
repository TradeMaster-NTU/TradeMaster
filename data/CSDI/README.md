# CSDI for Fintech Data Imputation

***
In order to provide an alternative solution of missing values in raw data rather that deleting them, we use the imputation model proposed in the following paper.

[CSDI: Conditional Score-based Diffusion Models for Probabilistic Time Series Imputation](https://arxiv.org/abs/2107.03502) *NeurIPS 2021*

## Usage
This algorithm supports missing data imputation of day level. Please make sure your raw data file is in csv format.   

The default data format requires six indicators: date, open, high, low, close, adjcp. Please make changes of line 9, 33 and 34 in [dataset_own.py](https://github.com/TradeMaster-NTU/TradeMaster/blob/main/data/CSDI/dataset_own.py) if you need to add, drop or change indicators.  

Step 1: Training and imputation of own dataset
   ```
   python exe_own.py --dataset [dataset name] --tic [ticker name]  --testmissingratio [missing ratio]
   ```
The default missing ratio is set as 0.1. Please do not edit it unless there are too many missing values in your own dataset.  


Step 2: Generation of new dataset
   ```
   python impute.py --dataset [dataset name] --tic [ticker name]  --testmissingratio [missing ratio]
   ```


Step 3: Visualization
   ```
   python visual.py --dataset [dataset name] --tic [ticker name]  --testmissingratio [missing ratio] --dataind [data index]
   ```
to visualize point chart of open, high, low, close and adjcp of 10-days interval as well as the candlestick chart

[visualize_example.ipynb](https://github.com/TradeMaster-NTU/TradeMaster/blob/main/visual_example.ipynb) is a notebook for directly visualizing results.



