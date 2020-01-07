# -*- coding: utf-8 -*-
# Make extensive use of https://github.com/makagan/InferenceGAN
"""
@author: Yoann Boget
"""


import matplotlib.pyplot as plt
import numpy as np
from CGAN import InferGAN


class argstuff(object):
    def __init__(self, epoch=10000, batch_size=1000, \
                 save_dir="models", result_dir="results", \
                dataset="toy0", log_dir="logs", \
                gpu_mode = False, gan_type="IGAN", \
                 noise_dist = "normal", \
                lrG = 0.0001, lrD = 0.0001, \
                 lr_decay_step = 10000, \
                beta1 = 0, beta2=0.9,
                gammaMSE=0):
        
        self.epoch = epoch
        self.batch_size=batch_size
        self.save_dir = save_dir
        self.result_dir = result_dir
        self.dataset = dataset
        self.log_dir = log_dir
        self.gpu_mode = gpu_mode
        self.gan_type = gan_type
        self.noise_dist = noise_dist
        self.lrG = lrG
        self.lrD = lrD
        self.lr_decay_step = lr_decay_step
        self.beta1 = beta1
        self.beta2 = beta2
        self.gammaMSE=gammaMSE
        
args = argstuff()
gan = InferGAN(args)
gan.train()
