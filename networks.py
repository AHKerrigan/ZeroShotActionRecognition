#from json import encoder
#from msilib.schema import Feature
from torch.nn import TransformerEncoder, TransformerEncoderLayer

#from positional_encodings import PositionalEncoding1D

from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange

import torch.nn.functional as F
import torchvision.models as models

import torch.nn as nn

import numpy as np

import torch

from transformers import ViTModel, DetrForSegmentation

import copy
import pickle

EMBED_SIZE = {"w2vavg" : 300,
              "w2vavg+dense" : 300,
              'masked_w2v_seq' : 300,
              "BERT" : 768,
              "sent2vec" : 700,
              "sent2vec+dense" : 700,
              "bertavg": 768,
              'CLIP' : 512,
              'clip+dense' : 512}

def get_network(opt):
    """
    Selection function for available networks.conda 
    """
    if opt.network == 'multiplication':
        return PSZSAR_Original(models.video.r2plus1d_18, hidden_dim=1024, embed=opt.text_embed, fixconvs=opt.fixconvs, pretrain=opt.pretrainbackbone)
    else:
        assert NotImplementedError


class SentFineTune(nn.Module):
    def __init__(self, hidden_dim=1024, class_embeds='sent2vec+dense'):
        super(SentFineTune, self).__init__()
        
        self.lay1 = nn.Linear(EMBED_SIZE[class_embeds], 1024)
        self.lay2 = nn.Linear(1024, 600)


    def forward(self, x):
        x = self.lay1(x)
        x = F.relu(x)
        x = self.lay2(x)
        x = F.normalize(x, dim=-1)
        return x

class PSZSAR_Original(nn.Module):
    
    def __init__(self, network, hidden_dim=1024, fixconvs=False, pretrain=False, embed='sent2vec'):
        super(PSZSAR_Original, self).__init__()

        self.model = network(pretrained=pretrain)
        if fixconvs:
            for param in self.model.parameters():
                param.requires_grad = False
                
        video_feature_dim = self.model.fc.in_features
        
        self.model = torch.nn.Sequential(*(list(self.model.children())[:-1]))
 
        self.fine_tuner = SentFineTune(class_embeds=embed)
        #self.fine_tuner = nn.Identity()

        
        self.dropout2 = torch.nn.Dropout(p=0.05)
        self.v_to_joint = nn.Linear(video_feature_dim, hidden_dim)
        
        self.t_to_joint = nn.Linear(600, hidden_dim)

        self.classification = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(True),
            nn.Dropout(p=0.1),
            nn.Linear(hidden_dim//2, 1)
        )
         
    def forward(self, x, class_embeds=None):
        bs, nc, ch, l, h, w = x.shape

        #print("class embed shape is", class_embeds.shape)
        class_embeds = class_embeds.squeeze(0)

        x = rearrange(x, 'bs cl ch f h w -> (bs cl) ch f h w')
        x = self.model(x)
        x_visual = reduce(x, '(bs cl) ch f h w -> bs ch', 'mean', bs=bs)

        #print("Class emebed shape is", class_embeds.shape)
        class_embeds = class_embeds.squeeze(1)
        #print("x after backbone is", x.shape)
         
        if class_embeds is None:
            print('Need to input class embeddings')
            assert NotImplementedError
        
        new_embeds = self.fine_tuner(class_embeds)
            
        x_dropout2 = self.dropout2(x_visual)
        
        b, _ = x_dropout2.shape
        c, _ = new_embeds.shape

        v_feats = F.normalize(self.v_to_joint(x_dropout2)).unsqueeze(1).repeat(1, c, 1)  # (B, F)
        t_feats = F.normalize(self.t_to_joint(new_embeds)).unsqueeze(0).repeat(b, 1, 1)  # (C, F)

        #print("v_feats and t_feats", v_feats.shape, t_feats.shape)
        
        combined_feats = v_feats*t_feats
        
        #print("combined feats shape is", combined_feats.shape)
        classification = self.classification(combined_feats).squeeze(-1)  # (B, C)
        
        #print("classification shape", classification.shape)
        return classification   

class Kinetics_Supervised(nn.Module):
    
    def __init__(self, network, hidden_dim=1024, fixconvs=True, nopretrained=False, embed='sent2vec'):
        super(Kinetics_Supervised, self).__init__()

        self.model = network(pretrained=True)
        if fixconvs:
            for param in self.model.parameters():
                param.requires_grad = False
                
        video_feature_dim = self.model.fc.in_features
        
        self.model = torch.nn.Sequential(*(list(self.model.children())[:-1]))
 
        self.fine_tuner = SentFineTune(class_embeds=embed)

        
        self.dropout2 = torch.nn.Dropout(p=0.05)
        self.v_to_joint = nn.Linear(video_feature_dim, hidden_dim)
        
        self.t_to_joint = nn.Linear(600, hidden_dim)

        self.classification = nn.Sequential(
            nn.Linear(512, hidden_dim//2),
            nn.ReLU(True),
            nn.Dropout(p=0.1),
            nn.Linear(hidden_dim//2, 330)
        )
         
    def forward(self, x, class_embeds=None):
        bs, nc, ch, l, h, w = x.shape

        #print("class embed shape is", class_embeds.shape)
        class_embeds = class_embeds.squeeze(0)
        x = rearrange(x, 'bs cl ch f h w -> (bs cl) ch f h w')
        x = self.model(x)
        x_visual = reduce(x, '(bs cl) ch f h w -> bs ch', 'mean', bs=bs)
        class_embeds = class_embeds.mean(1)
         
        if class_embeds is None:
            print('Need to input class embeddings')
            assert NotImplementedError
        
        
        #print("combined feats shape is", combined_feats.shape)
        classification = self.classification(x_visual)  # (B, C)
        
        return classification   