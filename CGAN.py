#!/usr/bin/env python
#making use of:
# https://github.com/znxlwm/pytorch-generative-model-collections
# https://github.com/wiseodd/generative-models/tree/master/GAN
# Make use of https://github.com/makagan/InferenceGAN
"""
@author: Yoann Boget
"""

import utils, torch, time, os, pickle, sys
import numpy as np
import torch as to
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from IPython import display
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats as st
from scipy.signal import medfilt


########################################################################
### Utilities
########################################################################
def true_X_given_Y(given_Y=0.5, model_number=3, slice_size=0.01, N=10000000):
    x, y = get_toy_samples(N, model_number=1)
    XY= np.concatenate((x, y), axis=1)
    X_given_Y_true = XY[(XY[:,1]>(given_Y-(slice_size/2)))*(XY[:,1]<(given_Y+(slice_size/2))), 0]
    return X_given_Y_true

def compute_distances(G_in, given_Y, noise_dist, n_samples=10000, n_test_samples=100000):
    #device = to.device("cuda" if to.cuda.is_available() else "cpu")
    device='cpu' 
    G_in.eval()
    z_ = get_random_inputs((n_samples,10), dist=noise_dist) 

    y_ = np.asarray(given_Y).reshape((1,1))
    y_ = to.from_numpy(y_).float().to(device).repeat(n_samples,1)
    x_gen = G_in(z_.to(device), y_)
    x_gen = x_gen.detach().cpu().numpy()
    true_x=true_X_given_Y(given_Y)
    nBins = 1000
    MyRange= [-0.5, 1.5]
    true_cond_dist,_,_ = plt.hist(true_x, bins=nBins,range=MyRange, density=True,histtype='step') 
    gen_cond_dist,_ ,_ = plt.hist(x_gen, bins=nBins,range=MyRange, density=True,histtype='step')
    plt.close()
    
    true_n_gen_cond= np.concatenate((np.expand_dims(true_cond_dist, axis=1),np.expand_dims(gen_cond_dist, axis=1)), axis=1)      
    true_n_gen_cond= true_n_gen_cond[true_n_gen_cond[:,1]!=0,:]       
                    
    KL= st.entropy(true_n_gen_cond[:,0], true_n_gen_cond[:,1])            
    m = 0.5 * (true_cond_dist+gen_cond_dist)
    JS= 0.5 * (st.entropy(true_cond_dist, m) + st.entropy(gen_cond_dist,m))
    return KL, JS             
          
def make_density_plot(G_in, given_Y, noise_dist, n_samples=10000, n_test_samples = 100000): 
    #device = to.device("cuda" if to.cuda.is_available() else "cpu")
    device='cpu'
    G_in.eval()
    z_ = get_random_inputs((n_samples,10), dist=noise_dist) 
    y_ = np.asarray(given_Y).reshape((1,1))
    y_ = to.from_numpy(y_).float().to(device).repeat(n_samples,1)

    nBins = 1000
    MyRange= [-0.5, 1.5]
    x_gen = G_in(z_.to(device), y_)
    x_gen = x_gen.detach().cpu().numpy()
    true_X=true_X_given_Y(given_Y)   
    sns.kdeplot(np.squeeze(x_gen), shade=True, color="r", bw=0.001)
    sns.kdeplot(np.squeeze(true_X), shade=True, color="g", bw=0.001)
    y_plt1, _, _=plt.hist(np.squeeze(true_X), bins=nBins,range=MyRange, density=True,histtype='step')
    y_plt2, _, _=plt.hist( np.squeeze(x_gen), bins=nBins,range=MyRange, density=True,histtype='step')

    plt.title('Approximated densities of $p_x(x|y='+ str(given_Y) +'$ (green) and $p_{g(z)}(x|y='+ str(given_Y) +')$ (red)')
    plt.ylabel('Density')
    plt.xlabel('x|(y = '+ str(given_Y) +')')
    plt.xlim(given_Y-0.5, 
                 given_Y+0.5)
    plt.ylim(0, max(max(y_plt1),max(y_plt2))+0.1*max(max(y_plt1),max(y_plt2)))
    plt.show()

    return

def make_distance_plot(dist_list):
    plt.plot(range(0, (len(dist_list))*10, 10), dist_list, markersize=1)
    plt.title('Distance')
    plt.xlabel('Updates of the generator')
    plt.legend(loc="upper right")
    plt.ylim(0, 0.5)
    plt.show()

########################################################################
################## Data sampling
########################################################################


def get_toy_samples(n, model_number=1):
    #device = to.device("cuda" if to.cuda.is_available() else "cpu")
    device='cpu'
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


def get_random_inputs(size, dist="uniform"):
    if dist=="uniform":
        return torch.rand(size)
    elif dist=="normal":
        return torch.randn(size)
    elif dist=="zeros":
        return torch.zeros(size)
    else:
        print("ERROR: get_random_inputs(...) must have dist=\"uniform\" or \"normal\" or \"zeros\" ")
        sys.exit(0)
        

########################################################################
################## Nets
########################################################################

class generator(nn.Module):
    def __init__(self):
        super(generator, self).__init__()

        self.noise_size = 10
        self.latent_size = 1
        self.input_dim = self.noise_size + self.latent_size
        self.output_dim = 1

        self.fc = nn.Sequential(
            nn.Linear(self.input_dim, 256),

            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, self.output_dim),
        )

        utils.initialize_weights(self)
            

    def forward(self, input, latent):
        x = torch.cat([input, latent], 1)
        x = self.fc(x)
        return x


class discriminator(nn.Module):
    def __init__(self):
        super(discriminator, self).__init__()

        self.input_size = 1
        self.latent_size = 1
        self.input_dim = self.input_size + self.latent_size
        self.output_dim = 1

        self.fc = nn.Sequential(
            nn.Linear(self.input_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, self.output_dim),
            nn.Sigmoid(),
        )

        utils.initialize_weights(self)

    def forward(self, input, latent):
        x = torch.cat([input, latent], 1)
        x = self.fc(x)
        return x


class InferGAN(object):
    def __init__(self, args):
        # parameters
        self.epoch = args.epoch
        self.batch_size = args.batch_size
        self.save_dir = args.save_dir
        self.result_dir = args.result_dir
        self.dataset = args.dataset
        self.log_dir = args.log_dir
        self.gpu_mode = args.gpu_mode
        self.model_name = args.gan_type
        self.noise_dist = args.noise_dist


        self.G = generator()
        self.D = discriminator()

        self.G_optimizer = optim.Adam(self.G.parameters(), lr=args.lrG, betas=(args.beta1, args.beta2))
        self.D_optimizer = optim.Adam(self.D.parameters(), lr=args.lrD, betas=(args.beta1, args.beta2))

        self.lrG = args.lrG
        self.lrD = args.lrD
        self.lr_decay_step = args.lr_decay_step
        

        if self.gpu_mode:
            self.G.cuda()
            self.D.cuda()
            self.BCE_loss = nn.BCELoss().cuda()
        else:
            self.BCE_loss = nn.BCELoss()

        print('---------- Networks architecture -------------')
        utils.print_network(self.G)
        utils.print_network(self.D)
        print('-----------------------------------------------')

        # load data

        self.data_X, self.data_Y = get_toy_samples(10000)
        self.z_dim = 10
        self.y_dim = 1

        #print("data X")
        #print(self.data_X)
        #print("data Y")
        #print(self.data_Y)

    def train(self):
        self.train_hist = {}
        self.train_hist['D_loss'] = []
        self.train_hist['G_loss'] = []
        self.train_hist['per_epoch_time'] = []
        self.train_hist['total_time'] = []
        self.train_hist['KL_dist']=[]
        self.train_hist['JS_dist']=[]
        

        if self.gpu_mode:
            self.y_real_, self.y_fake_ = Variable(torch.ones(self.batch_size//2, 1).cuda()), Variable(torch.zeros(self.batch_size//2, 1).cuda())
        else:
            self.y_real_, self.y_fake_ = Variable(torch.ones(self.batch_size//2, 1)), Variable(torch.zeros(self.batch_size//2, 1))

        self.D.train()
        print('training start!!')
        start_time = time.time()
        n_update_gen = 5
        for epoch in range(self.epoch):
            self.G.train()
            epoch_start_time = time.time()

            #Learning rate decay
            if self.lr_decay_step is not None:
                if epoch % self.lr_decay_step == 0 and epoch > 0:
                    self.lrG *= 0.5
                    for param_group in self.G_optimizer.param_groups:
                        param_group['lr'] = self.lrG
                    self.lrD *= 0.5
                    for param_group in self.D_optimizer.param_groups:
                        param_group['lr'] = self.lrD
                    

            perm_ = torch.randperm(len(self.data_X))
            data_X_perm_ = self.data_X[perm_]
            data_Y_perm_ = self.data_Y[perm_]
            
            for iter in range(len(self.data_X) // self.batch_size):
                x_ = data_X_perm_[iter*self.batch_size:iter*self.batch_size + self.batch_size//2]
                y_ = data_Y_perm_[iter*self.batch_size:iter*self.batch_size + self.batch_size//2]


                x_gen_ = data_X_perm_[iter*self.batch_size + self.batch_size//2:(iter+1)*self.batch_size]
                y_gen_ = data_Y_perm_[iter*self.batch_size + self.batch_size//2:(iter+1)*self.batch_size]
                #z_ = torch.rand((len(y_gen_), self.z_dim))
                z_ = get_random_inputs( (len(y_gen_), self.z_dim), dist=self.noise_dist)

                if self.gpu_mode:
                    x_, z_, y_, x_gen_, y_gen_ = Variable(x_.cuda()), Variable(z_.cuda()), Variable(y_.cuda()), Variable(x_gen_.cuda()), Variable(y_gen_.cuda())
                else:
                    x_, z_, y_, x_gen_, y_gen_ = Variable(x_), Variable(z_), Variable(y_), Variable(x_gen_), Variable(y_gen_)

                if ((iter+1) % n_update_gen != 0):
                    # update D network
                    self.D.train()
                    self.G.eval()
                    self.D_optimizer.zero_grad()

                    D_real = self.D(x_, y_)
                    D_real_loss = self.BCE_loss(D_real, self.y_real_)

                    G_ = self.G(z_, x_gen_)
                    D_fake = self.D(x_gen_, G_)
                    D_fake_loss = self.BCE_loss(D_fake, self.y_fake_)

                    D_loss = D_real_loss + D_fake_loss
                    self.train_hist['D_loss'].append(D_loss.data)

                    D_loss.backward()
                    self.D_optimizer.step()

                else:
                    # update G network
                    self.G.train()
                    self.D.eval()
                    self.G_optimizer.zero_grad()

                    G_ = self.G(z_, x_)
                    D_fake = self.D(x_, G_)
                    G_loss = self.BCE_loss(D_fake, self.y_real_)
                    self.train_hist['G_loss'].append(G_loss.data)

                    G_loss.backward()
                    self.G_optimizer.step()

                if ((iter + 1) % 10) == 0:
                    print("Epoch: [%2d] [%4d/%4d] D_loss: %.8f, G_loss: %.8f" %
                          ((epoch + 1), (iter + 1), len(self.data_X) // self.batch_size, D_loss.data, G_loss.data))

                # if((iter + 1) % 500) == 0:
                #     xD = range(len(self.train_hist['D_loss']))
                #     xG = np.array(range(len(self.train_hist['G_loss'])))
                #     xG = xG*(n_update_gen-1)
                
                #     display.clear_output(wait=True)
                #     display.display(plt.gcf())
                    
                #     plt.plot(xD, medfilt(self.train_hist['D_loss'],3), label="loss D")
                #     plt.plot(xG, medfilt(self.train_hist['G_loss'],3), label="loss G")
                #     plt.legend()
                #     plt.show()
                    
                #     make_density_plot(self.G, given_Y=0.1, noise_dist=self.noise_dist, gpu_mode=self.gpu_mode)
                #     make_density_plot(self.G, given_Y=0.5, noise_dist=self.noise_dist, gpu_mode=self.gpu_mode)
                #     make_density_plot(self.G, given_Y=0.9, noise_dist=self.noise_dist, gpu_mode=self.gpu_mode)
                #     #self.G.cuda()
                #     self.G.train()


                    
            if ((epoch+1) % 50) == 0:
                
                xD = range(len(self.train_hist['D_loss']))
                xG = np.array(range(len(self.train_hist['G_loss'])))
                xG = xG*(n_update_gen-1)

                display.clear_output(wait=True)
                display.display(plt.gcf())

                plt.plot(xD, medfilt(self.train_hist['D_loss'],11), label="loss D")
                plt.plot(xG, medfilt(self.train_hist['G_loss'],11), label="loss G")
                plt.legend()
                plt.show()
                
                KL_sum, JS_sum= 0, 0
                for y in [0.1, 0.5, 0.9]:
                    KL, JS= compute_distances(self.G, given_Y=y, noise_dist=self.noise_dist)
                    make_density_plot(self.G, given_Y=y, noise_dist=self.noise_dist)
                    KL_sum, JS_sum = KL_sum+KL, JS_sum+JS
                self.train_hist['KL_dist'].append(KL_sum)                
                self.train_hist['JS_dist'].append(JS_sum)
                make_distance_plot(self.train_hist['KL_dist'])
                make_distance_plot(self.train_hist['JS_dist'])
                
            
        
                print('KL divergence: {KL}' .format(KL=KL_sum))
                print('JS divergence: {JS}' .format(JS=JS_sum))
                    #self.G.cuda()
                self.G.train()


            self.train_hist['per_epoch_time'].append(time.time() - epoch_start_time)
            #self.visualize_results((epoch+1))

            
        self.train_hist['total_time'].append(time.time() - start_time)
        print("Avg one epoch time: %.2f, total %d epochs time: %.2f" % (np.mean(self.train_hist['per_epoch_time']),
              self.epoch, self.train_hist['total_time'][0]))
        print("Training finish!... save training results")

        self.save()
        #utils.generate_animation(self.result_dir + '/' + self.dataset + '/' + self.model_name + '/' + self.model_name, self.epoch)
        #utils.loss_plot(self.train_hist, os.path.join(self.save_dir, self.dataset, self.model_name), self.model_name)


    def save(self):
        save_dir = os.path.join(self.save_dir, self.dataset, self.model_name)

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        torch.save(self.G.state_dict(), os.path.join(save_dir, self.model_name + '_G.pkl'))
        torch.save(self.D.state_dict(), os.path.join(save_dir, self.model_name + '_D.pkl'))

        with open(os.path.join(save_dir, self.model_name + '_history.pkl'), 'wb') as f:
            pickle.dump(self.train_hist, f)

    def load(self):
        save_dir = os.path.join(self.save_dir, self.dataset, self.model_name)

        self.G.load_state_dict(torch.load(os.path.join(save_dir, self.model_name + '_G.pkl')))
        self.D.load_state_dict(torch.load(os.path.join(save_dir, self.model_name + '_D.pkl')))
                
        
        


