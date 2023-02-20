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
<br />
   ```
   python tools/MarketRegimeLabeling/Label.py --data_path data/algorithmic_trading/BTC/test.csv --method linear --fitting_parameters 2/7 2/14 4 --labeling_parameters -0.5 0.5
   ```

DJ30
    
    python tools/MarketRegimeLabeling/Label.py --data_path data/portfolio_management/dj30/test.csv --method linear --fitting_parameters 2/7 2/14 4 --labeling_parameters -0.25 0.25 --regime_number 3 --length_limit 24
    
for DJ30 applied in PM use-case, we would like to define the market regime based on DJ30 index. We have provided an example of
test_labeled_3_24.csv which is DJI_labeled_3_24.csv and test.csv merged on 'date' where  DJI_labeled_3_24.csv is got from running 

DJI index
   ```
   python tools/MarketRegimeLabeling/Label.py --data_path  data/portfolio_management/dj30/DJI.csv --method linear --fitting_parameters 2/7 2/14 4 --labeling_parameters -0.25 0.25  --regime_number 3 --length_limit 24  --PM data/portfolio_management/dj30/test.csv
   ```

BTC & PD_BTC

    python tools/MarketRegimeLabeling/Label.py --data_path data/algorithmic_trading/BTC/test.csv --method linear --fitting_parameters 2/7 2/14 4 --labeling_parameters -0.15 0.15 --regime_number 3 --length_limit 24

    python tools/MarketRegimeLabeling/Label.py --data_path data/order_execution/PD_BTC/test.csv --method linear --fitting_parameters 2/7 2/14 4 --labeling_parameters -0.15 0.15 --regime_number 3 --length_limit 24

OE_BTC

    python tools/MarketRegimeLabeling/Label.py --data_path data/order_execution/BTC/test.csv --method linear --fitting_parameters 2/7 2/14 4 --labeling_parameters -0.01 0.01 --regime_number 3 --length_limit 32 --OE_BTC True

Exchange

     python tools/MarketRegimeLabeling/Label.py --data_path data/portfolio_management/exchange/test.csv --method linear --fitting_parameters 2/7 2/14 4 --labeling_parameters -0.05 0.05 --regime_number 3 --length_limit 24


<br />
You may read the comments labeling_util.py to get a hint on how to set the parameters

The script will take in a data file and output the file with a market regime label column. Besides the market label, we also provide a stock group label column based on DWT clustering.

## Testing agent under a specific market regime

Please prepare test data with the instruction in Usage and run agent with additional args, for example

```
 python tools/algorithmic_trading/train.py --task_name style_test --test_style 0
```