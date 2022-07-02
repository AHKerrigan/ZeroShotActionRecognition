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
    'architecture' : opt.archname
}

wandb.init(project='zsar', 
        entity='ahkerrigan',
        config=config)
wandb.run.name = opt.description
wandb.save()


# Get the datasets
if opt.dataset == 'kinetics':
    dataloaders = dataloader.get_kinetics_ucf_hmbd(opt)

    kinetics_dataset = dataloaders['training'][0]
    train_dataloader = torch.utils.data.DataLoader(kinetics_dataset, batch_size=opt.batch_size, num_workers=0, shuffle=True, drop_last=False)
    
    ucf_dataset = dataloaders['testing'][0]
    val_dataloader1 = torch.utils.data.DataLoader(ucf_dataset, batch_size=opt.batch_size, num_workers=0, shuffle=True, drop_last=False)
    
    hmdb_dataset = dataloaders['testing'][1]
    val_dataloader2 = torch.utils.data.DataLoader(hmdb_dataset, batch_size=opt.batch_size, num_workers=0, shuffle=True, drop_last=False)

# Cross entropy based training 
if opt.traintype == 'ce':   
    criterion = torch.nn.CrossEntropyLoss()
    train_one_epoch = train.train_one_epoch_pws

model = networks.get_network(opt)

optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)

scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [60, 120], gamma=0.1)



_ = model.to(opt.device)
wandb.watch(model, criterion, log="all")

acc10 = 0
for epoch in range(opt.n_epochs): 

    if not opt.evaluate:
        _ = model.train()
        train_one_epoch(train_dataloader, val_dataloader1, model, criterion, optimizer, scheduler, opt, epoch, val_dataloader2)


    scheduler.step()
    
    
    #acc10 = max(acc10, validate_one_epoch(val_dataloader, model, opt, epoch, writer))

    #print("Best acc10 is", acc10)
    #validate_loss(val_dataloader, model, opt, epoch, writer)

