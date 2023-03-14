# Publication
[Deep Reinforcement Learning for Quantitative Trading: Challenges and Opportunities](https://ieeexplore.ieee.org/abstract/document/9779600) *(IEEE Intelligent Systems 2022)*

[DeepScalper: A Risk-Aware Reinforcement Learning Framework to Capture Fleeting Intraday Trading Opportunities](https://arxiv.org/abs/2201.09058) *(CIKM 2022)*

[Commission Fee is not Enough: A Hierarchical Reinforced Framework for Portfolio Management](https://ojs.aaai.org/index.php/AAAI/article/view/16142) *(AAAI 21)*

[Reinforcement Learning for Quantitative Trading (Survey)](https://dl.acm.org/doi/10.1145/3582560) *(ACM TIST 2023)*

[PRUDEX-Compass: Towards Systematic Evaluation of Reinforcement Learning in Financial Markets](https://arxiv.org/abs/2302.00586)

# File Structure
Here is the structure of the TradeMaster project. 

```
| TradeMaster
| ├── configs
| ├── data
| │   ├── algorithmic_trading 
| │   ├── high_frequency_trading  
| │   ├── order_excution          
| │   └── porfolio_management
| ├── deploy
| │   ├── backend_client.py
| │   ├── backend_client_test.py 
| │   └── backend_service.py        
| │   ├── backend_service_test.py  
| ├── docs
| ├── figure
| ├── installation
| │   ├── docker.md
| │   ├── requirements.md
| ├── tools
| │   ├── algorithmic_trading          
| │   ├── data_preprocessor
| │   ├── high_frequency_trading
| │   ├── market_dynamics_labeling
| │   ├── missing_value_imputation  
| │   ├── order_excution  
| │   ├── porfolio_management  
| │   ├── __init__.py      
| ├── tradmaster       
| │   ├── agents   
| │   ├── datasets 
| │   ├── enviornments 
| │   ├── evaluation 
| │   ├── imputation 
| │   ├── losses
| │   ├── nets
| │   ├── preprocessor
| │   ├── optimizers
| │   ├── pretrained
| │   ├── trainers
| │   ├── transition
| │   ├── utils
| │   └── __init__.py     
| ├── unit_testing
| ├── Dockerfile
| ├── LICENSE
| ├── README.md
| ├── pyproject.toml
| └── requirements.txt
```




