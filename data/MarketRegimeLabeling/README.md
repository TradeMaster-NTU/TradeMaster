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

It is recommended to run through the example.ipynb notebook to visualize the labeling process. This will also give hints on
deciding the parameters for your dataset.

And example of labeling the data
   ```
   python Label.py --data_path ../data/dj30/test.csv --method linear --fitting_parameters 2/7 2/14 4 --labeling_parameters -0.5 0.5
   ```
You may read the comments labeling_util.py to get a hint on how to set the parameters

