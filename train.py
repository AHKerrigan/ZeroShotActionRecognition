import time
from scipy.spatial.distance import cdist
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

from colorama import Fore, Style

from einops import rearrange

import torch
import torch.nn.functional as F
import pickle
from tqdm import tqdm

import wandb
import evaluate
import pandas as pd



def train_one_epoch_pws(train_dataloader, val_dataloader1,  model, criterion, optimizer, scheduler, opt, epoch, val_dataloader2=None):

    loss_cycle = (len(train_dataloader.dataset.data) // (opt.batch_size * opt.loss_per_epoch))
    val_cycle = len(train_dataloader.dataset.data) // (opt.batch_size * opt.eval_per_epoch)


    print("Outputting loss every", val_cycle, "batches")
    print("Validating every", val_cycle*5, "batches")
    print("Starting Epoch", epoch)

    bar = tqdm(enumerate(train_dataloader), total=len(train_dataloader), disable=opt.cluster)

    # Get the information about the class embed dimensions
    n_classes, class_seq_len, class_embed_dim = train_dataloader.dataset.class_embeds.shape

    # Send class embeds to GPU
    class_embeds = torch.from_numpy(train_dataloader.dataset.class_embeds).to(opt.device).unsqueeze(0).repeat(opt.gpus, 1, 1, 1)
    totalloss = 0
    acc1_1 = 0
    acc1_5 = 0

    for i ,(vids, labels) in bar: 

        vids = vids.to(opt.device)
        labels = labels.to(opt.device)
        
        outs = model(vids, class_embeds)
        #print(outs.shape)

        #print("Ours and labels", outs.shape, labels.shape)
        loss = criterion(outs, labels)
        totalloss += loss.item()

        bar.set_description(f'Loss: {round(totalloss / (i + 1), 4)} - {opt.valset1} Top-1 {round(acc1_1, 4)} - Top-5 {round(acc1_5, 4)}')

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

        '''
        if i % loss_cycle == 0:
            wandb.log({opt.dataset + " " + opt.traintype + " Classification Loss" : loss.item()})
        if i % val_cycle == 0:
            if val_dataloader1:
                acc1_1, acc1_5 = evaluate.evaluate_pws(val_dataloader1, model, opt)
                wandb.log({opt.valset1 + " " + opt.traintype + " Top-1 Accuracy" : acc1_1})
                wandb.log({opt.valset1 + " " + opt.traintype + " Top-5 Accuracy" : acc1_5})
            if val_dataloader2:
                acc2_1, acc2_5 = evaluate.evaluate_pws(val_dataloader2, model, opt)
                wandb.log({opt.valset2 + " " + opt.traintype + " Top-1 Accuracy" : acc2_1})
                wandb.log({opt.valset2 + " " + opt.traintype + " Top-5 Accuracy" : acc2_5})
        '''

