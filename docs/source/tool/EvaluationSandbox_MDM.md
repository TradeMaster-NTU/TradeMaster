# Evaluation Sandbox: Market Dynamics Modeling

## Introduction 
This part of evaluation sandbox provides a tool for user to model market dynamics on datasets and evaluate their policy under the market of specific market dynamic. 
The sandbox shows visualizations and reports to assist user compare policies across market dynamic.

## Market Dynamics Modeling
The Market Dynamics modeling is a module to label raw data with dynamics that is interpretable and controllable. Users may tune the hyperparameters to get the dynamics that are most suitable for their policy.
The dynamics can be used as meta-information. For example, in the evaluation process, user can run evaluation on specific dynamics.

## Usage 
- You may use the to label the dataset with modeled dynamics.
  1. Configure the parameters in [`market_dynamics_modeling.py`](../../../configs/market_dynamics_modeling/market_dynamics_modeling.py), **you may refer to[`TradeMaster_Sandbox_whitepaper.pdf`](TradeMaster_Sandbox_whitepaper.pdf) to get further information on the parameters and modeling algorithm.** 
  1. Run the [`run.py`](../../../tools/market_dynamics_labeling/run.py)  in tools/market_dynamics_labeling to prepare the dataset. You may refer to the experiment logs and the result visualizations(metrics_of_each_dynamics.png and f"slice_and_merge_model_{dynamic_number}dynamics_minlength{min_length_limit}_{labeling_method}_labeling_{tic}.png") to tune the parameters until you are satisfied with the result.
  ```
  $ python tools/market_dynamics_labeling/run.py
  ```

- After you have labeled the dataset with market dynamics, you may want to test your agent under specific market dynamics.
  1. **Update the 'test_style_path'** in the config files to the labeled dataset path you get from previous step. The default test_style_path is the path of and example dataset (which is not labeled by this tool). 
  2. Run the trainer with arguments `--task dynamics_test --test_dynamic dynamic_label` to perform evaluation on specific market dynamic. You will get reports and visualization result.
  ```
  $ python tools/algorithmic_trading/train.py --task_name dynamics_test --test_dynamic 0
  ```
  <div align="center">
          <img src="example_figs/Radar_plot.png" width = 400 height =  />
        </div> 
  The visualization result is a radar plot different metrics, while the 'Metric Radar' polygon is the score of metrics, the 'Profitability' is a mean score of the upper 4 metrics while the "Risk_Control" is the mean of the lower 2 metrics. Noted that the metrics various between different tasks.
#### Parameters 
please check [`TradeMaster_Sandbox_whitepaper.pdf`](TradeMaster_Sandbox_whitepaper.pdf) for details on the parameters. We have provided default parameters in the config file [`market_dynamics_modeling.py`](../../../configs/market_dynamics_modeling/market_dynamics_modeling.py) and the [`base`](../../../configs/_base_/market_dynamics_model) folder. Noted that these parameters are just a starting point for you to play with. As each person have his or her own expectation and usage on market dynamics, please tune the parameters and run experiments to get better results.
#### Scoring
The scores of the visualization result are calculated as described:
- Do nothing metrics are used as score 0
- Blind Buy metrics are used as score 50 (-50 if worse than Do Nothing)
- The score of other agents are given based on the assumption that the scores are following a normal distribution (50,$\sqrt{50}$)
##### Baselines
  - Buy and Hold: This is and ideal policy where you spend all your cash on the first tick.
  - Blind Buy: Buy as much as possible until the cash runs out.
  - Do Nothing: Take no action at all
## Example
We show a use case of using the tool to label dynamics on an [`example BTCUSDT dataset`](https://datasets.tardis.dev/v1/binance-futures/book_snapshot_5/2020/09/01/BTCUSDT.csv.gz)(click to download) from the open example of a data provider [`Tardis.dev`](https://docs.tardis.dev/downloadable-csv-files)   
We have already aggregate the data to second level and the data is provided [`here`](../../../data/market_dynamics_modeling/second_level_BTC_LOB/data.feather)
With the [`configuration file`](../../../configs/market_dynamics_modeling/market_dynamics_modeling.py), you will get the following dynamics modeling result:
  <div align="center">
          <img src="example_figs/Radar_plot.png" width = 400 height =  />
        </div> 






## Try Online API
Check our [online platform](http://trademaster.ai/) for more information.


