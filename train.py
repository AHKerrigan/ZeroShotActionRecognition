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

from avalanche.benchmarks.generators import nc_benchmark

def train_one_epoch_pws(train_dataloader, val_dls,  model, criterion, optimizer, opt, epoch, n_samples, class_embeds):

    loss_cycle = n_samples // (opt.batch_size * opt.loss_per_epoch)
    val_cycle = n_samples // (opt.batch_size * opt.eval_per_epoch)

    print("Outputting loss every", loss_cycle, "batches")
    print("Validating every", val_cycle, "batches")
    print("Starting Epoch", epoch)

    bar = tqdm(enumerate(train_dataloader), total=len(train_dataloader), disable=opt.cluster)

    # Get the information about the class embed dimensions
    n_classes, class_seq_len, class_embed_dim = class_embeds.shape

    # Send class embeds to GPU
    class_embeds = torch.from_numpy(class_embeds).to(opt.device).unsqueeze(0).repeat(opt.gpus, 1, 1, 1)
    totalloss = 0
    totaltrainacc1 = 0
    totaltrainacc5 = 0
    acc1_1 = 0
    acc1_5 = 0

    for i ,(d) in bar: 

        vids = d[0].to(opt.device)
        labels = d[1].to(opt.device)
        
        outs = model(vids, class_embeds)
        #print(outs.shape)

        #print("Ours and labels", outs.shape, labels.shape)
        loss = criterion(outs, labels)

        outs = outs.cpu().detach().numpy()
        labels = labels.cpu().detach().numpy()

        temptrainacc1, temptrainacc5 = evaluate.compute_accuracy_prob(outs, labels)
        totaltrainacc1 += temptrainacc1
        totaltrainacc5 += temptrainacc5

        totalloss += loss.item()

        bar.set_description(f'Loss: {round(totalloss / (i + 1), 3)} - {opt.valset1} Top-1 {round(acc1_1, 3)}-Top-5{round(acc1_5, 3)}- {opt.dataset} Top-1 {round(totaltrainacc1 / (i + 1), 3)} Top-5 {round(totaltrainacc5 / (i + 1), 3)}')

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

        if i % loss_cycle == 0:
            wandb.log({opt.dataset + " " + opt.traintype + " Classification Loss" : loss.item()})
            wandb.log({opt.dataset + " " + opt.traintype + " Train Top-1 Accuracy" : temptrainacc1})
            wandb.log({opt.dataset + " " + opt.traintype + " Train Top-5 Accuracy" : temptrainacc5})
        if (i + 1) % val_cycle == 0:
            for DL in val_dls:
                acc1_1, acc1_5 = evaluate.evaluate_pws(DL, model, opt)
                if opt.wandb: wandb.log({DL.dataset.name + " " + opt.traintype + " Top-1 Accuracy" : acc1_1})
                if opt.wandb: wandb.log({DL.dataset.name + " " + opt.traintype + " Top-5 Accuracy" : acc1_5})

def regular_loop(train_one_epoch, datasets,  model, criterion, optimizer, scheduler, opt):
    wandb.watch(model, criterion, log="all")
    
    train_dataloader = torch.utils.data.DataLoader(datasets['training'][0], batch_size=opt.batch_size, num_workers=opt.processes, shuffle=True, drop_last=False)

    val_dls = []
    for val_ds in datasets['testing']:
        val_dls.append(torch.utils.data.DataLoader(val_ds, batch_size=opt.batch_size, num_workers=opt.processes, shuffle=True, drop_last=False))
    
    class_embeds = datasets['training'][0].class_embeds
    n_samples = len(train_dataloader.dataset.data)
    for epoch in range(opt.n_epochs): 

        if not opt.evaluate:
            _ = model.train()
            train_one_epoch(train_dataloader, val_dls, model, criterion, optimizer, opt, epoch, n_samples, class_embeds)
        torch.save(model.state_dict(), f'weights/{opt.description}_{opt.dataset}_{opt.network}.pth')

        scheduler.step()


# Basic training loop for a class incremental task 
def incremental_loop(train_one_epoch, datasets,  model, criterion, optimizer, scheduler, opt):

    wandb.watch(model, criterion, log="all")

    # Split the training dataset based on tasks 
    scenerio = nc_benchmark(datasets['training'][0], datasets['training'][0], n_experiences=10, shuffle=True, seed=1234, task_labels=False)

    val_dls = []
    # Create the validation dataloaders 
    for val_ds in datasets['testing']:
        val_dls.append(torch.utils.data.DataLoader(val_ds, batch_size=opt.batch_size, num_workers=opt.processes, shuffle=True, drop_last=False))
    
    class_embeds = datasets['training'][0].class_embeds
    for experience in scenerio.train_stream:

        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, opt.step_size, gamma=0.5, last_epoch=- 1, verbose=False)
        
        print("Starting experience", experience.task_label)
        train_dataloader = torch.utils.data.DataLoader(experience.dataset, batch_size=opt.batch_size, num_workers=opt.processes, shuffle=True, drop_last=False)

        n_samples = len(experience.dataset)
        for epoch in range(opt.n_epochs): 

            if not opt.evaluate:
                _ = model.train()
                train_one_epoch(train_dataloader, val_dls, model, criterion, optimizer, opt, epoch, n_samples, class_embeds)
            torch.save(model.state_dict(), f'weights/{opt.description}_{opt.dataset}_{opt.network}.pth')

            scheduler.step()


