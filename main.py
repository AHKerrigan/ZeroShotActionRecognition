import os, numpy as np, argparse, time, multiprocessing
from tqdm import tqdm

import torch
import torch.nn as nn

#import network
import dataloader
import train

#from torch.utils.tensorboard import SummaryWriter
import wandb

from torch.nn import TransformerEncoder, TransformerEncoderLayer
#from networks import GeoCLIP, VGGTriplet, BasicNetVLAD
import networks 
from config import getopt

opt = getopt()

config = {
    'learning_rate' : opt.lr,
    'epochs' : opt.n_epochs,
    'batch_size' : opt.batch_size,
}

if opt.wandb:
    wandb.init(project='zsar', 
            entity='ahkerrigan',
            config=config)
    wandb.run.name = opt.description
    wandb.save()


# Get the datasets
if opt.dataset == 'kinetics':
    datasets = dataloader.get_kinetics_ucf_hmbd(opt)
    
# Doing the incremental task or not 
if opt.incremental:
    loop = train.incremental_loop
else:
    loop = train.regular_loop

# Cross entropy or nearest neighbor based training 
if opt.traintype == 'ce':   
    criterion = torch.nn.CrossEntropyLoss()
    train_one_epoch = train.train_one_epoch_pws

_ = criterion.to(opt.device)
model = networks.get_network(opt)

optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)

scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [60, 120], gamma=0.1)



_ = model.to(opt.device)
wandb.watch(model, criterion, log="all")

acc10 = 0
loop(train_one_epoch, datasets, model, criterion, optimizer, scheduler, opt)
