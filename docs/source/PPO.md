# Proximal Policy Optimization(PPO)

## 1.Background
 
 0) Policy Gradient and Trust Region Policy Optimization

$$ 
    L^{PG}= \hat{E}_t [\nabla log \pi (a_t|s_t) A_t]
$$

Updating Policy Gradient without Constrains may lead to destructive large gradient update. 

$$
    Maxmize \ \ \hat{E}_t [\frac{\pi_\theta (a_t|s_t)} {\pi_{\theta old} (a_t|s_t)}\hat{A}_t-\beta KL [\pi_{\theta old} (a_t|s_t),\pi_\theta (a_t|s_t)]]
$$
 
However, The choice of  *beta is based on heuristics thus hard to decide for multiple problems.

## 2. Viarants
  
  1) Adaptive KL Penalty

$$
    Maxmize \ \ \hat{E}_t [\frac{\pi_\theta (a_t|s_t)}{\pi_{\theta old}(a_t|s_t)}\hat{A}_t-\beta KL [\pi_{\theta old} (a_t|s_t),\pi_\theta (a_t|s_t)]]
$$

$$ 
    Compute \ \ d = \hat{E}_t[KL [\pi_{\theta old} (a_t|s_t),\pi_\theta (a_t|s_t)]]
$$
 
$$
    if \ \ d < d_{targ}/1.5, \beta \leftarrow \beta / 2
$$

$$
    if \ \ d > d_{targ}/1.5, \beta \leftarrow \beta * 2
$$

  2) Clipped Surrogate
 
$$
    L^{CLIP} (\theta) = \hat{E}_t[min(r_t (\theta)) \hat{A}_t, clip(r_t (\theta) , 1-epsilon, 1+epsilon) \hat{A}_t]
$$


## 3.Traning

  Recurrent Neural Network (RNN) is used to recursively update Policy Gradient.
  

