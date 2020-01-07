# -*- coding: utf-8 -*-
"""
Éditeur de Spyder

Ceci est un script temporaire.
"""
import numpy as np
import torch as to
    
### MODELS  ####
def model(model_number, n, device):
    np.random.seed(123)
    x=np.random.rand(n, 1)
    a=1
    b=20
    eps = np.random.randn(n, 1)*0.05
    if model_number==1:
        #model 1
        y=a*x+eps

    elif model_number==2: #model 2 heteroscedasticity
        eps2= np.random.randn(n, 1)*(a*x)*0.1
        y=a*x+eps2
    elif model_number==3:
    #model 3 
        y=a*x+0.2*np.sin(b*x)+eps

    y=to.from_numpy(y).float().to(device)
    x=to.from_numpy(x).float().to(device)
    return x, y


def true_X_given_Y(given_Y, device, model_number=3, slice_size=0.01, N=10000000):
    x=np.random.rand(N, 1)
    a=1
    b=20
    eps = np.random.randn(N, 1)*0.05
    if model_number== 1:
        #model 1
        y=a*x+eps

    elif model_number==2: #model 2 heteroscedasticity
        eps2= np.random.randn(N, 1)*(a*x)*0.1
        y=a*x+eps2
    elif model_number==3:
    #model 3 
        y=a*x+0.2*np.sin(b*x)+eps
    XY= np.concatenate((x, y), axis=1)
    X_given_Y_true = XY[(XY[:,1]>(given_Y-(slice_size/2)))*(XY[:,1]<(given_Y+(slice_size/2))), 0]
    return X_given_Y_true

def generated_X_given_Y(given_Y, model, N_generated, device, Ones_gen, noise_dim):
    noise_gen = to.from_numpy(np.random.randn(N_generated, noise_dim)).type(to.FloatTensor).to(device)
    given_Y_vect=Ones_gen*given_Y
    X_given_Y_generated=model(given_Y_vect, noise_gen)

    return X_given_Y_generated.detach().cpu().numpy()

### Misc ####
def zeros_and_ones(size, device):
    Ones =to.ones([size, 1]).to(device)
    Zeros =to.zeros([size, 1]).to(device) 
    return Zeros, Ones

#### Metrics #####
def init_var(given_Y):
     KL=[[] for x in range(len(given_Y))]      
     JS=[[] for x in range(len(given_Y))]
     KS=[[] for x in range(len(given_Y))]
     mean_dist=[[] for x in range(len(given_Y))]
     losses=[[],[]]
     saved_g_loss, saved_d_loss=losses
     generated_mean= [[] for x in range(len(given_Y))] 
     generated_var=[[] for x in range(len(given_Y))]  
     generated_skew=[[] for x in range(len(given_Y))]  
     generated_kurt=[[] for x in range(len(given_Y))]
     return KL, JS, KS, mean_dist, losses, saved_g_loss, saved_d_loss, generated_mean, generated_var, generated_skew, generated_kurt

def Variables_declaration(given_Y, device, model_number, model_name, ones):
    X_given_Y_true=[None]*len(given_Y)
    X_given_Y_generated=[None]*len(given_Y)
    true_meanX_Y=[None]*len(given_Y)
    N_generated=[None]*len(given_Y)
    noise_gen=[None]*len(given_Y)
    Ones_gen=[None]*len(given_Y)
    Metrics = init_var(given_Y)           
    nBins = 1000
    MyRange= [-0.5, 1.5]       
    for j in range(len(given_Y)):
        X_given_Y_true[j]=true_X_given_Y(given_Y[j], device, model_number=model_number, slice_size=0.01, N=10000000)
        true_meanX_Y[j]=np.mean(X_given_Y_true[j])
        N_generated[j]= len(X_given_Y_true[j])
        noise_gen[j]=to.Tensor(np.random.rand(N_generated[j], 1)).to(device)    
        Ones_gen[j]= to.ones([N_generated[j], 1]).to(device)
    save_true= X_given_Y_true, true_meanX_Y, N_generated
    to.save(save_true, model_name+'/save_true')
    return  save_true,  Metrics, X_given_Y_generated, Ones_gen, nBins, MyRange
