# Evaluation Sandbox: Market Dynamics Modeling

## Introduction 
The evaluation sandbox provides a sandbox for user to evaluate their policy under different scenarios . 
The sandbox shows visualizations and reports to assist user compare policies across market dynamic.

## Market Dynamics Modeling
The Market Dynamics modeling is a module to label raw data with dynamics that is interpretable. 
The dynamics are used as meta-information. For example, in the evaluation process, user can run evaluation on specific dynamics.

## Usage & Example
This module prepare data for evaluation, to run a full test you should follow this pipeline:
- Run the [`run.py`]()  in tools/market_dynamics_labeling or [`run_linear_model.py`]() to prepare the dataset
  1. Tune the parameters based on the visualization results
     <div align="center">
       <img src="example_figs/dm_result_1.png" width = 400 height =  />
     </div>
  1. Increase `length_limit`
      <div align="center">
       <img src="example_figs/dm_result_2.png" width = 400 height =  />
      </div>
  1. Modify `labeling_parameters`
      <div align="center">
        <img src="example_figs/dm_result_3.png" width = 400 height =  />
      </div> 
- Update the 'test_style_path' in the config files to the dataset path you get from previous step.

- Run the trainer with arguments `--task dynamics_test --test_dynamic dynamic_label` to perform evaluation on specific market dynamic. You will get reports and visualization result.
  <div align="center">
          <img src="example_figs/Radar_plot.png" width = 400 height =  />
        </div> 
#### Parameters 
- `fitting_parameters`: This is a set of parameters for the filter, please refer to the comment in lines for detailed description. 
- `labeling_parameters`: This is a set of parameters for dynamic classification, please refer to the comment in lines for detailed description. 
- `dynamic_number`: This is the number of dynamics. 
- `length_limit`: This is the minimum length of a consecutive time-series of same dynamic. 

#### Scoring
The scores of the visualization result are calculated as described:
- Do nothing metrics are used as score 0
- Blind Buy metrics are used as score 50 (-50 if worse than Do Nothing)
- The score of other agents are given based on the assumption that the scores are following a normal distribution (50,$\sqrt{50}$)
##### Baselines
  - Buy and Hold: This is and ideal policy where you spend all your cash on the first tick.
  - Blind Buy: Continues buy until the cash runs out.
  - Do Nothing: Do nothing




## Examples
### Use market dynamics model to prepare evaluation datasets
It is recommended to run through the trademaster/evaluation/market_dynamics_labeling/example.ipynb notebook to visualize the labeling process. This will also give hints on
deciding the parameters for your dataset. The example.html contains the visualization results from example.ipynb.
#### Running from configuration file 
Change the parameters in `configs/evaluation/market_dynamics_modeling.py` and run 
```
$ python tools/market_dynamics_labeling/run.py
```

### Testing agent under a specific market dynamic
```
$ python tools/algorithmic_trading/train.py --task_name dynamics_test --test_dynamic 0
```

## Try out the pipeline online 
Check our [online platform](http://trademaster.ai/) for more information.


