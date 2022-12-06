# Market Regime Labeling

***
Market regime can be a useful market feature, but it is not well-defined. The intuition of this model is to label the time-series data
into different regimes. 


## Methods

### Linear Model
Using slope of linear regression model as the metric for market regime, this is the most explainable
method where each regime has an explicit threshold.

### Markov Regime Switching Model(*under construction)
The Markov regime switching model assume there are multiple hidden state and use a markov model to accommodate time series with state-dependent parameters.<br /> 
[Hamilton, J. D. (2010). Regime switching models. In Macroeconometrics and time series analysis (pp. 202-209). Palgrave Macmillan, London.](https://link.springer.com/chapter/10.1057/9780230280830_23) presents an example of
markov regime switching model.
While it has the ability to explore hidden states, the model is less explainable and controllable. We only provide an example of how this kind of model works and
do not apply it to The pipeline.


## Usage

It is recommended to run through the example.ipynb notebook to visulize the labeling process. This will also give hints on
deciding the parameters for your dataset.

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

[visualize_example.ipynb](https://github.com/TradeMaster-NTU/TradeMaster/blob/main/data/CSDI/visual_example.ipynb) is a notebook for directly visualizing results.



