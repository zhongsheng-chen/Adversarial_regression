# Adversarial_regression

Conditional Progressive Generative Adversarial Networks with Multiple Encoders
A Method for Image In-painting and Image Extension
 
Master Thesis: Yoann Boget
Supervisor at SLAC: Micheal Kagan
Supervisor at University of Neuchâtel: Clément Chevalier

Abstract

Adversarial Regression is a proposition to perform high dimensional non-linear regression with uncertainty estimation. It uses Conditional Generative Adversarial Network to obtain an estimate of the full predictive distribution for a new observation. 

Generative Adversarial Networks (GAN) are implicit generative models which produce samples from a distribution approximating the distribution of the data. The conditional version of it can be expressed as follow: $\min\limits_G \max\limits_D V(D, G) = \mathbb{E}_{\bm{x}\sim p_{r}(\bm{x})} [log(D(\bm{x}, \bm{y}))] + \mathbb{E}_{\bm{z}\sim p_{\bm{z}}(\bm{z})} [log (1-D(G(\bm{z}, \bm{y})))]$, where D and G are real-valuated functions, $\bm{x}$ and $\bm{y}$ respectively the explained and explanatory variables, and $\bm{z}$ a noise vector from a known distribution, typically the standard normal. An approximate solution can be found by training simultaneously two neural networks to model D and G and feeding G with $\bm{z}$. After training, we have that $G(\bm{z}, \bm{y})\mathrel{\dot\sim} p_{data}(\bm{x}, \bm{y})$. By fixing $y$, we have $G(\bm{z}|\bm{y}) \mathrel{\dot\sim} p_{data}(\bm{x}|\bm{y})$. By sampling $\bm{z}$, we can therefore obtain samples following approximately $p(\bm{x}|\bm{y})$, which is the predictive distribution of $\bm{x}$ for a new $\bm{y}$. 

We ran experiments to test various loss functions, data distributions, sample sizes, and dimensions of the noise vector. Even if we observed differences, no set of hyperparameters consistently outperformed  the others. The quality of CGAN for regression relies on fine-tuned hyperparameters depending on the task rather than on fixed values. From a broader perspective, the results show that adversarial regressions are promising methods to perform uncertainty estimation for high dimensional non-linear regression. 

