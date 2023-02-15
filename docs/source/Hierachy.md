# Directory Hierachy

Here is the structure of the TradeMaster project. 

```
| TradeMaster
| ├── configs
| │   ├── base
| │   ├── algorithmic_trading
| │   ├── order_excution
| │   └── porfolio_management
| ├── data
| │   ├── algorithmic_trading          
| │   ├── order_excution          
| │   └──  porfolio_management
| ├── deploy
| │   ├── backend_client_test.py         
| │   ├── backend_client.py
| │   ├── backend_service_test.py  
| │   └── backend_service.py  
| ├── tools
| │   ├── algorithmic_trading          
| │   ├── MarketRegimeLabeling   
| │   ├── order_excution  
| │   ├── porfolio_management  
| │   ├── __init__.py 
| │   └── tmp.py      
| ├── tradmaster
| │   ├── __pycache         
| │   ├── agents   
| │   ├── datasets 
| │   ├── enviornments 
| │   ├── losses
| │   ├── nets
| │   ├── optimizers
| │   ├── pretrained
| │   ├── trainers
| │   ├── utils
| │   └── __init__.py     
| ├── unit_testing
| │   ├── test_agents\algorithmic_trading        
| │   ├── test_datasets
| │   ├── test_enviornments 
| │   ├── test_losses
| │   ├── test_nets
| │   ├── test_optimizers
| │   ├── test_trainers
| │   ├── __init__.py   
| │   └── test_score.py  
| ├── LICENSE
| ├── python3.9.yaml
| └── README.md
```

In the folder structure above:

- ``configs`` contains configuration files directory for agents trainng.
- ``data`` contains the datasets directory for training and testing.
- ``deploy`` contains the datasets for training and testing.
- ``tools`` contains training scripts.
- ``trademaster`` contains the files defining agents,enviornments, training losses, nets, optimizers and helper functions
- ``unit_testing`` contains the components for testing.
































<!-- 
| TradeMaster
| ├── configs
| │   ├── base
| │   ├── algorithmic_trading
          └── algorithmic_trading_BTC_dqn_dqn_adam_mse.py
| │   ├── order_excution
          ├── order_execution_BTC_eteo_eteo_adam_mse.py
          └── order_execution_BTC_pd_pd_adam_mse.py
| │   ├── porfolio_management
| | |        ├── portfolio_management_dj30_deeptrader_deeptrader_adam_mse.py
          ├── portfolio_management_dj30_eiie_eiie_adam_mse.py
          ├── portfolio_management_dj30_investor_imitator_investor_imitator_adam_mse.py
          ├── portfolio_management_dj30_sarl_sarl_adam_mse.py
          ├── portfolio_management_exchange_a2c_a2c_adam_mse.py
          ├── portfolio_management_exchange_ddpg_ddpg_adam_mse.py
          ├── portfolio_management_exchange_pg_pg_adam_mse.py
          ├── portfolio_management_exchange_ppo_ppo_adam_mse.py
          ├── portfolio_management_exchange_sac_sac_adam_mse.py
          └── portfolio_management_exchange_td3_td3_adam_mse.py
|     └── __init__.py
| ├── data
| │   ├── algorithmic_trading          
| │   ├── order_excution          
| │   └──  porfolio_management
| ├── deploy
| │   ├── backend_client_test.py         
| │   ├── backend_client.py
| │   ├── backend_service_test.py  
| │   └── backend_service.py  
| ├── tools
| │   ├── algorithmic_trading          
| │   ├── MarketRegimeLabeling   
| │   ├── order_excution  
| │   ├── porfolio_management  
| │   ├── __init__.py 
| │   └── tmp.py      
| ├── tradmaster
| │   ├── __pycache         
| │   ├── agents   
| │   ├── datasets 
| │   ├── enviornments 
| │   ├── losses
| │   ├── nets
| │   ├── optimizers
| │   ├── pretrained
| │   ├── trainers
| │   ├── utils
| │   └── __init__.py     
| ├── unit_testing
| │   ├── test_agents\algorithmic_trading        
| │   ├── test_datasets
| │   ├── test_enviornments 
| │   ├── test_losses
| │   ├── test_nets
| │   ├── test_optimizers
| │   ├── test_trainers
| │   ├── __init__.py   
| │   └── test_score.py  
| ├── LICENSE
| ├── python3.9.yaml
| └── README.md -->


