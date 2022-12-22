# Market Regime Labeling

***
Market regime can be a useful market feature, but it is not well-defined. The intuition of this model is to label the time-series data
into different regimes. 


## Methods

### Linear Model
Using slope of linear regression model as the metric for market regime, this is the most explainable
method where each regime has an explicit threshold.

#### parameters 
1. method: We only provide linear method for now
2. fitting_parameters: This is a set of parameters for the filter, please refer to the comment in lines for detailed description.
3. labeling_parameters: This is a set of parameters for regime classification, please refer to the comment in lines for detailed description.
4. regime_number: This is the number of regimes.
5. length_limit: This is the minimum length of a consecutive time-series of same regime. 

## Usage

It is recommended to run through the example.ipynb notebook to visualize the labeling process. This will also give hints on
deciding the parameters for your dataset. The example.html contains the visualization results from example.ipynb.

An example of labeling the data
   ```
   python Label.py --data_path ../data/dj30/test.csv --method linear --fitting_parameters 2/7 2/14 4 --labeling_parameters -0.5 0.5
   ```

An example of labeling the data for stylized-TimeGan
    ``
    python Label.py --data_path ../data/dj30/test.csv --method linear --fitting_parameters 2/7 2/14 4 --labeling_parameters -0.25 0.25 --regime_number 3 --length_limit 24
    ``
You may read the comments labeling_util.py to get a hint on how to set the parameters

The script will take in a data file and output the file with a market regime label column. Besides the market label, we also provide a stock group label column based on DWT clustering.

