# Adversarial_regression
## Generative Adversarial Networks for Non-Linear Regression: Theory and Assessment
 
Master Thesis: Yoann Boget  
Supervisor at SLAC (Stanford University: Dr. Micheal Kagan  
Supervisor at University of Neuchâtel: Dr. Clément Chevalier  
https://arxiv.org/abs/1910.09106  
Grade: 5.5/6 (Very good)

### Abstract

Adversarial Regression is a proposition to perform high dimensional non-linear regression with uncertainty estimation. It uses Conditional Generative Adversarial Network to obtain an estimate of the full predictive distribution for a new observation. 

Generative Adversarial Networks (GAN) are implicit generative models which produce samples from a distribution approximating the distribution of the data. The conditional version of it can be expressed as follow: 

<a href="https://www.codecogs.com/eqnedit.php?latex=\min\limits_G&space;\max\limits_D&space;V(D,&space;G)&space;=&space;\mathbb{E}_{\bm{x}\sim&space;p_{r}(\bm{x})}&space;[log(D(\bm{x}|&space;\bm{y}))]&space;&plus;&space;\mathbb{E}_{\bm{z}\sim&space;p_{\bm{z}}(\bm{z})}&space;[log&space;(1-D(G(\bm{z}|&space;\bm{y}&space;))]" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\min\limits_G&space;\max\limits_D&space;V(D,&space;G)&space;=&space;\mathbb{E}_{\bm{x}\sim&space;p_{r}(\bm{x})}&space;[log(D(\bm{x}|&space;\bm{y}))]&space;&plus;&space;\mathbb{E}_{\bm{z}\sim&space;p_{\bm{z}}(\bm{z})}&space;[log&space;(1-D(G(\bm{z}|&space;\bm{y}&space;))]" title="\min\limits_G \max\limits_D V(D, G) = \mathbb{E}_{\bm{x}\sim p_{r}(\bm{x})} [log(D(\bm{x}| \bm{y}))] + \mathbb{E}_{\bm{z}\sim p_{\bm{z}}(\bm{z})} [log (1-D(G(\bm{z}| \bm{y} ))]" /></a>

where _D_ and _G_ are real-valuated functions, ___x___ and ___y___ respectively the explained and explanatory variables, and ___z___ a noise vector from a known distribution, typically the standard normal. An approximated solution can be found by training simultaneously two neural networks to model _D_ and _G_. After training, we have that _G(__z__, __y__)_ approximate _p(__x__, __y__)_. By fixing ___y___, we have _G(__z__|__y__)_ approximating _p(__x__|__y__)_. By sampling ___z___, we can therefore obtain samples following approximately _p(__x__|__y__)_, which is the predictive distribution of ___x___ for a new observation ___y___. 

We ran experiments to test various loss functions, data distributions, sample sizes, and dimensions of the noise vector. Even if we observed differences, no set of hyperparameters consistently outperformed  the others. The quality of CGAN for regression relies on fine-tuned hyperparameters depending on the task rather than on fixed values. From a broader perspective, the results show that adversarial regressions are promising methods to perform uncertainty estimation for high dimensional non-linear regression. 

