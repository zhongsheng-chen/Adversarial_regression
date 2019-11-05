# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 18:59:27 2019

@author: yboget
"""

########################################################
##### Preprocessing and Variable Definition ############

import torch as to
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from scipy import stats as st
import numpy as np
import argparse
import os
import shutil
import metrics

p = argparse.ArgumentParser()
p.add_argument("gan_type", type=str, help="Either SGAN, WGAN-clip or WGAN-GP")
p.add_argument("model_number", type=int, help="Integer: 1, 2 or 3")
p.add_argument("n", type=int, help="Number of events")
p.add_argument("batch_size", type=int, help="Batch_size: should be a multiple of ")
p.add_argument("d_steps", type=int, help="Number of discriminator update(s) by epoch")
p.add_argument("g_steps", type=int, help="Number of generator update(s) by epoch")
p.add_argument("lr", type=float, help="Learning rate")
p.add_argument("epochs", type=int, help="Number of epochs")
p.add_argument("decreasing_LR", type=bool, help="Decreasing learning rate")
p.add_argument("rate_LR", type=float, help="Rate of decrease for the learning rate")
p.add_argument("save_interval", type=int, help="Number of epochs")
p.add_argument("noise_dim", type=int, help="Dimension of noise Z")
p.add_argument("beta1", type=float, help="Parameter for Adam")
p.add_argument("beta2", type=float, help="Parameter for Adam")
p.add_argument("--name_comp", type=str, help="Add something to the name", default='')
p.add_argument("--restart_epoch", type=int, help="Add something to the name", default=0)
p.add_argument("--add_noise", type=bool, help="If yes add noise to the input")
args = p.parse_args()


device = to.device("cuda" if to.cuda.is_available() else "cpu")
print(device)
# Model parameters
gan_type= args.gan_type
model_number = args.model_number
n = args.n 
batch_size = args.batch_size
d_steps = args.d_steps
g_steps = args.g_steps 
lr= args.lr
epochs= args.epochs
decreasing_LR= args.decreasing_LR
rate_LR= args.rate_LR
save_interval= args.save_interval
name_comp= args.name_comp
noise_dim= args.noise_dim
beta1= args.beta1
beta2= args.beta2
add_noise= args.add_noise
lambd=0.1


increasing_DS=False
lambd=0.1
iterations =510000

################################
#####  Networks ################

# Model Definition: Generator 3 layers - relu
class Generator(nn.Module):
    def __init__(self, noise_dim):
        super(Generator, self).__init__()
        noise_dim = noise_dim
        latent_dim = 1
        input_dim = noise_dim + latent_dim
        output_dim = 1        
        
        self.GeneratorNN = to.nn.Sequential(


          
            nn.Linear(input_dim, 256),
            
            nn.ReLU(),
            nn.Linear(256, 128),
            
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
            
        )
    def forward(self, input, latent):
        x = to.cat([input, latent], 1)
        x = self.GeneratorNN(x)
        return x
# Critic: 4 laysers (3 relu - linear)
class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        input_dim=2
        output_dim=1
        self.CriticNN = to.nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, output_dim),
            nn.Sigmoid(),
        )

    def forward(self, x):
        y_pred= self.CriticNN(x)
        return y_pred

        
# Discriminator: 4 laysers (3 relu - sigmoid)
class Discriminator(nn.Module):
    def __init__(self,):
        super(Discriminator, self).__init__()
        input_dim = 2
        output_dim = 1
        self.DiscriminatorNN = to.nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 256),
            
            nn.LeakyReLU(0.2),
            nn.Linear(256, output_dim),
            nn.Sigmoid(),
        )
    def forward(self, x):
        y_pred= self.DiscriminatorNN(x)
        return y_pred


######################################
#### GAN FUNCTIONS ###################
        

def gradient_penalty(model, real_data, fake_data, batch_size, lambd, device, Ones):   
    alpha = to.from_numpy(np.random.rand(batch_size, 1)).type(to.FloatTensor).to(device)
    xhat = alpha * real_data + ((1 - alpha) * fake_data)
    xhat = to.autograd.Variable(xhat, requires_grad=True).to(device)    
    C_xhat = model(xhat).to(device)   
    gradients = to.autograd.grad(outputs=C_xhat, inputs=xhat, grad_outputs=Ones,
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradient_penalty = to.mean((to.norm(gradients, 2, dim=1)-1)**2)*lambd
    return gradient_penalty



def GAN_training(x, y, n, model_D, model_G, d_optimizer, g_optimizer, lr, batch_size, 
                 iterations, model_number, device,
                 model_name, gan_type, d_steps, g_steps, lambd, 
                 noise_dim, add_noise, epoch_change=None, 
                 decreasing_LR=False, rate_LR=1, given_Y=[0.1, 0.4, 0.7, 1], restart_epoch=0):
       
    Xd= to.stack([x]*d_steps).reshape([n*d_steps, 1])
    Yd= to.stack([y]*d_steps).reshape([n*d_steps, 1])
    Xg= to.stack([x]*g_steps).reshape([n*g_steps, 1])
    Yg= to.stack([y]*g_steps).reshape([n*g_steps, 1])
    Zeros, Ones= metrics.zeros_and_ones(batch_size, device)
    epochs =np.int( np.ceil(iterations*(batch_size/n) + 10000 * (batch_size/n) ))
    counter=0
    save_true,  Metrics, X_given_Y_generated, Ones_gen, nBins, MyRange = metrics.Variables_declaration(given_Y, device, model_number, model_name, Ones)

    for epoch in range(epochs):
        print('start epoch: '+str(epoch))
        permutation_d = to.randperm(n*d_steps)
        permutation_g = to.randperm(n*g_steps)
        
        for k in range(int(n/batch_size)):        
            for d in range(d_steps): 
                                 
                indices = permutation_d[(k*d_steps*batch_size)+(d*batch_size):(k*d_steps*batch_size)+(d+1)*batch_size]                
                batch_x = Xd[indices, :]
                batch_y = Yd[indices, :]        
                
                model_D.train()
                model_G.eval()
                # batch_z = to.rand([batch_size, 1])
                d_optimizer.zero_grad()
                # Train D on real
                input_real = to.cat([batch_x, batch_y], dim=1)
                output_Dr = model_D(input_real)
                # Train D on fake
                noise = to.from_numpy(np.random.randn(batch_size, noise_dim)).type(to.FloatTensor).to(device)
                if add_noise:
                    batch_y=batch_y+ to.from_numpy(np.random.randn(batch_size, noise_dim)).type(to.FloatTensor).to(device)*0.005 
                      
                x_fake = model_G(batch_y, noise)
                output_Df = model_D(to.cat((x_fake, batch_y), dim=1)) 
                              
                if gan_type == 'SGAN':            
                    d_real_loss = criterion(output_Dr, Ones)
                    d_fake_loss = criterion(output_Df, Zeros) 
                    d_loss= d_real_loss+d_fake_loss
                elif gan_type == 'WGAN-GP':
                    GP=gradient_penalty(model_D, input_real, x_fake, batch_size, lambd, device,  Ones)
                    d_loss= to.mean(output_Df) - to.mean(output_Dr) + GP 
                elif gan_type == 'WGAN-clip':
                    d_loss= to.mean(output_Df) - to.mean(output_Dr)
                elif gan_type == 'RSGAN':
                    d_loss = -to.mean(to.log(to.sigmoid(output_Dr-output_Df)))
                elif gan_type == 'RaSGAN':
                    Da_real = to.sigmoid(output_Dr-to.mean(output_Df))
                    Da_fake = to.sigmoid(output_Df-to.mean(output_Dr))
                    d_loss = -to.mean(to.log(Da_real))-to.mean(to.log(1-Da_fake))
                    
                elif gan_type == 'LSGAN':
                    d_real_loss = to.mean((output_Dr-Ones)**2)
                    d_fake_loss = to.mean((output_Df-Zeros)**2)
                    d_loss= d_real_loss+d_fake_loss
                elif gan_type == 'RLSGAN':
                    d_loss = to.mean((output_Dr-output_Df)**2)
                d_loss.backward()
                d_optimizer.step()
                if gan_type == 'WGAN-clip':
                    for p in model_D.parameters():
                        p.data.clamp_(-0.01, 0.01)
                
            
            
            for g in range(g_steps):
                # Train G 
                model_G.train()
                model_D.eval()
                g_optimizer.zero_grad()
                indices = permutation_g[(k*g_steps*batch_size)+(g*batch_size):(k*g_steps*batch_size)+(g+1)*batch_size]
                batch_x = Xg[indices, :]
                batch_y = Yg[indices, :]
                noise = to.from_numpy(np.random.randn(batch_size, noise_dim)).type(to.FloatTensor).to(device)
                if add_noise:
                    batch_y=batch_y+ to.from_numpy(np.random.randn(batch_size, noise_dim)).type(to.FloatTensor).to(device)*0.005 
 
                
                x_fake_g = model_G(batch_y, noise)
                
                output_g = model_D(to.cat((x_fake_g, batch_y), dim=1)) 
                
                
                input_realg = to.cat([batch_x, batch_y], dim=1)
                output_Drg = model_D(input_realg)
                
                if gan_type == 'SGAN':
                    g_loss = criterion(output_g, Ones) 
                elif gan_type == 'RSGAN':
                    g_loss = -to.mean(to.log(to.sigmoid(output_g-output_Drg)))
                elif gan_type == 'RaSGAN':
                    
                    Da_real = to.sigmoid(output_Drg-to.mean(output_g))
                    Da_fake = to.sigmoid(output_g-to.mean(output_Drg))
                    g_loss = -to.mean(to.log(Da_fake))-to.mean(to.log(1-Da_real))
 


                elif gan_type== 'LSGAN':   
                    g_loss = to.mean((output_g-Ones)**2)
                    
                elif gan_type == 'RLSGAN':
                    g_loss = to.mean((output_g-output_Drg)**2)
                else:
                    g_loss = -to.mean(output_g)
                g_loss.backward()
                g_optimizer.step()
                counter=counter+1
            X_given_Y_true, true_meanX_Y, N_generated = save_true
            KL, JS, KS, mean_dist, losses, saved_g_loss, saved_d_loss, generated_mean, generated_var, generated_skew, generated_kurt=Metrics
            if (counter==0 or (counter) % 100 == 0):
                for j in range(len(given_Y)):
                    X_given_Y_generated[j] = metrics.generated_X_given_Y(given_Y[j], model_G, N_generated[j], 
                                       device, Ones_gen[j], noise_dim)            
                        
                    pXY_true, _ , _ = plt.hist(X_given_Y_true[j], bins=nBins,range=MyRange, density=True,histtype='step')       
                    pXY_generated, _ , _ = plt.hist(X_given_Y_generated[j], bins=nBins,range=MyRange, density=True,histtype='step')
            
                    pXY= np.concatenate((np.expand_dims(pXY_true, axis=1),np.expand_dims(pXY_generated, axis=1)), axis=1)      
                    pXY= pXY[pXY[:,1]!=0,:]       
                    
                    KL[j].append(st.entropy(pXY[:,0], pXY[:,1]))            
                    m = 0.5 * (pXY_true +pXY_generated)
                    JS[j].append(0.5 * ( st.entropy(pXY_true, m) + st.entropy(pXY_generated,m)))             
                    KS[j].append(st.ks_2samp(np.squeeze(X_given_Y_true[j]), np.squeeze(X_given_Y_generated[j]))[0])
                     
                    mean_dist[j].append(np.absolute(true_meanX_Y[j]-np.mean(X_given_Y_generated[j])))
                    generated_mean[j].append(np.mean(X_given_Y_generated[j]))
                    generated_var[j].append(np.var(X_given_Y_generated[j], ddof=1))
                    generated_skew[j].append(st.skew(X_given_Y_generated[j]))
                    generated_kurt[j].append(st.kurtosis(X_given_Y_generated[j]))
                
                saved_g_loss.append(g_loss)
                saved_d_loss.append(d_loss)
                losses= [saved_g_loss, saved_d_loss]
            
            if (counter == 0 or (counter) % 10000 == 0):
                to.save(model_G.state_dict(), model_name+'/G_iter'+str(counter)+'.pt')
                to.save(model_D.state_dict(), model_name+'/D_iter'+str(counter)+'.pt')               
                training_history_iter = JS, KL, KS, mean_dist, losses, generated_mean, generated_var, generated_skew, generated_kurt
                to.save(training_history_iter, model_name+'/training_history_iter'+str(counter))
                KL, JS, KS, mean_dist, losses, saved_g_loss, saved_d_loss, generated_mean, generated_var, generated_skew, generated_kurt = metrics.var_init(given_Y)      


            if decreasing_LR==True:
                decrease=1/rate_LR
                rate=decrease**(1/(epochs-epoch))
                lr= lr*rate
                d_optimizer = optim.Adam(model_D.parameters(), lr=lr, betas=(0, 0.9))
                g_optimizer = optim.Adam(model_G.parameters(), lr=lr, betas=(0, 0.9))
            
      
    return model_G, model_D, training_history_iter


model_name=(str(gan_type)+'_M'+ str(model_number)+'_n'+str(n)+'_BS'+str(batch_size)+ 
'_D'+str(d_steps)+ '_G'+str(g_steps)+'_LR'+str(lr)+ '_Dlr'+str(decreasing_LR)+
'_Ndim'+str(noise_dim)+'_B1'+str(beta1)+'_B2'+str(beta2)+ str(name_comp)+ str(add_noise))


if os.path.exists(model_name+'/'):
    shutil.rmtree(model_name+'/')
os.makedirs(model_name+'/')
         
 
                
model_G = Generator(noise_dim).to(device)
if gan_type=='SGAN':
    model_D = Discriminator().to(device)
else: 
    model_D = Critic().to(device)
                
criterion = nn.BCELoss()  # Binary cross entropy
d_optimizer = optim.Adam(model_D.parameters(), lr=lr, betas=(beta1, beta2))
g_optimizer = optim.Adam(model_G.parameters(), lr=lr, betas=(beta1, beta2))
x, y= metrics.model(model_number, n, device)


  
GAN_training(x, y, n, model_D, model_G, d_optimizer, g_optimizer, lr, batch_size, 
    iterations, model_number, device,
    model_name, gan_type, d_steps, g_steps, 
    lambd, noise_dim, add_noise, epoch_change=None, 
    decreasing_LR=decreasing_LR, rate_LR=rate_LR, given_Y=[0.1, 0.4, 0.7, 1])







