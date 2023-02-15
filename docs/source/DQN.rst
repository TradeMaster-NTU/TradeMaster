Deep Q-Net (DQN)
===========
| Contents:
| `1. Data`_
| `2. Agent`_
| `3. Evaluation`_
| `4. Quick-run Tutorial`_


1.	Data
---------
Bitcoin Data of 2989 day:

- Basic Info:
   tic(currency),date,high,low,open,close,
- technical Factors: 
   adjcp(adjusted close), open (relative open price to t-1), zhigh, zlow, zadjcp, zclose, zd_5 (5-day moving average), zd_10, zd_15, zd_20, zd_25, zd_30.
- Train-Validation-Test Split:0.8,0.1,0.1.
      Train: BTC 2393 days. Validation: 300 days. Test: 298 days

2.	Agent
------------
DQN:
^^^^^^^^^
Use Multi-layer Perceptron (MLP) as Q-net to approximate the function of action value Q_value.

Update Process:
^^^^^^^^^^^^^^^^
- Update Process
      *Q(s,a)<-- Q(s,a)+α(Rt+1+γV(St+1)-V(St))*
      
      *TD Target: Rt+1+γV(St+1)*

      *TD Error：Rt+1+γV(St+1)-V(St)*  
 
 - where 
   
   Q(s,a) is the action-value function,
         
   Rt is the reward at timestep t,
 
   γ is the discount factor,
      
   V(St) is the expected return for State at timestep t.

- Optimizer: Adam 
- Loss: MSE(Q_values-Q_labels), where *Q_labels=Ri+γ * Q_next * (1-done)*.

* Alogrithm ::
   
   * Initialize network and replay buffer
   * for t in timestep:
      * Act At, get reward Rt, update state St+1
      * Store <St,At,Rt,St+1> to Replay Buffer
      * If enough samples in Replay Buffer:
         * take N samples <Si,Ai,Ri,Si+1>
      * Caculate TD-target yi= Ri + γ Maxa Q(St+1, A)
      * Minimize Loss function L = sum i to N (yi-Q(Si))^2
      * Update Q-net
   * End for 

Replay Buffer:
^^^^^^^^^^^^^^^
Put <State,Action,Reward,Next_Q> in the memory replay buffer. 

-	Random Sampling satisfies the Markov Property.
-	Increase Data Efficiency since each sample could be used multiple times.

3.	Evaluation
--------------
- Profit Margin:  
   *LastDayAsset/FirstDayAsset-1* * *100%*

- Risk Criteria: 
        
        sharpe_ratio = np.mean(daily_return) / (np.std(daily_return) * (len(df) ** 0.5) + 1e-10)
        
        Volatility = np.std(daily_return)
        
        Max Draw Down = max((max(df["total assets"]) - df["total assets"]) / (max(df["total assets"])) + 1e-10)
        
        Calmar Ratio = np.sum(daily_return) / (mdd + 1e-10)
        
        Sortino Ratio = np.sum(daily_return) / (np.std(neg_ret_lst) + 1e-10) / (np.sqrt(len(daily_return))+1e-10)
        
4. Quick-run Tutorial
------------------------
- A jupyter notebook for a quick start for DQN. 
      Please `Click Here <https://github.com/yonggang-Xie/readthedocs/blob/main/docs/source/Quick_Run_DQN_for_Algorithm_Trading_on_BTC.ipynb>`_ 
