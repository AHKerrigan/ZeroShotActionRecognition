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

def compute_accuracy_prob(pred_prob, y):
    y_pred = pred_prob.argsort(1)
    accuracy = accuracy_score(y, y_pred[:, -1]) * 100
    accuracy_top5 = np.mean([l in p for l, p in zip(y, y_pred[:, -5:])]) * 100
    return accuracy, accuracy_top5

def evaluate_pws(test_dataloader, model, opt):


    # Get the information about the class embed dimensions
    n_classes, class_seq_len, class_embed_dim = test_dataloader.dataset.class_embeds.shape

    # Send class embeds to GPU
    class_embeds = torch.from_numpy(test_dataloader.dataset.class_embeds).to(opt.device).unsqueeze(0).repeat(opt.gpus, 1, 1, 1)    

    _ = model.eval()
    with torch.no_grad():
        ### For all test images, extract features
        n_samples = len(test_dataloader.dataset)


        preds = []
        targets = []

        bar = tqdm(enumerate(test_dataloader), total=len(test_dataloader), disable=opt.cluster)
        fi = 0
        
        for idx, (vids, labels) in bar:

            vids = vids.to(opt.device)
            
            outs = model(vids, class_embeds).cpu().detach().numpy()

            preds.append(outs)
            targets.append(labels)
        
        preds = np.concatenate(preds, axis=0)
        targets = np.concatenate(targets, axis=0)

    accuracy, accuracy_top5 = compute_accuracy_prob(preds, targets)

    return accuracy, accuracy_top5