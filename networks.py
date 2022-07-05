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
from positional_encodings.torch_encodings import PositionalEncoding1D, PositionalEncoding2D, PositionalEncoding3D, Summer


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
    if opt.network == 'trans1':
        return Trans1(models.resnet18, hidden_dim=1024, embed=opt.text_embed, fixconvs=opt.fixconvs, pretrain=opt.pretrainbackbone)
    if opt.network == 'resnet18frames':
        return ResNet18Frames(models.resnet18, hidden_dim=1024, embed=opt.text_embed, fixconvs=opt.fixconvs, pretrain=opt.pretrainbackbone)
    else:
        assert NotImplementedError

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x): 

        qkv = self.to_qkv(x).chunk(3, dim = -1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class CustomAttention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_k = nn.Linear(dim, inner_dim, bias = False)
        self.to_v = nn.Linear(dim, inner_dim, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, q, k, v):
        
        q = self.to_q(q)
        k = self.to_k(k)
        v = self.to_v(v)

        qkv = (q, k, v)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class CustomTransformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.dim = dim

        self.lnk = nn.LayerNorm(dim)
        self.lnv = nn.LayerNorm(dim)
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                nn.LayerNorm(dim),
                CustomAttention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, q, k, v):
        k = self.lnk(k)
        v = self.lnv(v)
        for ln1, attn, ff in self.layers:
            q = ln1(q)
            x = attn(q, k, v) + q
            x = ff(x) + x
        return x


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

class PStranslayer(nn.Module):
    def __init__(self):
        super(PStranslayer, self).__init__()

        self.video_self = Transformer(768, 1, 12, 64, 1024)
        self.vid_project = nn.Linear(768, 768)
        
        self.cross = CustomTransformer(769, 1, 12, 64, 1024)
    def foward(self, vid, class_embeds):

        x = self.video_self(vid)
        x_p = self.vid_project(x)
        c_p = self.cross(class_embeds, x, x)

class ResNet18Frames(nn.Module):
    def __init__(self, network, hidden_dim=1024, fixconvs=False, pretrain=False, embed='sent2vec'):
        super(ResNet18Frames, self).__init__()

        self.backbone = network(pretrained=False)
        self.cls = nn.Linear(1000, 350)
    def forward(self, x, class_embeds):
        bs, nc, ch, f, h, w = x.shape
        x = x.squeeze(1)
        x = rearrange(x, 'bs ch f h w -> (bs f) ch h w')
        x = self.backbone(x)
        x = rearrange(x, '(bs f) dim -> bs f dim', bs=bs, f=f).mean(1)

        x = self.cls(x)
        return x


class Trans1(nn.Module):
    
    def __init__(self, network, hidden_dim=1024, fixconvs=False, pretrain=False, embed='sent2vec'):
        super(Trans1, self).__init__()

        self.model = network(pretrained=pretrain)
        if fixconvs:
            for param in self.model.parameters():
                param.requires_grad = False
                
        video_feature_dim = self.model.fc.in_features
        
        self.model = torch.nn.Sequential(*(list(self.model.children())[:-1]))

        self.project_v = nn.Linear(512, 768)
        self.dropout2 = torch.nn.Dropout(p=0.05)

        self.cls = torch.rand(768).cuda()

        self.transformer = Transformer(768, 12, 12, 64, 1024)

        self.pos = Summer(PositionalEncoding1D(768))

        self.classification = nn.Sequential(
            nn.Linear(768, hidden_dim//2),
            nn.ReLU(True),
            nn.Dropout(p=0.1),
            nn.Linear(hidden_dim//2, 350)
        )
         
    def forward(self, x, text):
        bs, nc, ch, f, h, w = x.shape

        x = rearrange(x, 'bs nc ch f h w -> (bs nc f) ch h w')
        x = self.model(x)
        x_visual = reduce(x, '(bs nc f) ch h w -> (bs nc) f ch', 'mean', bs=bs, nc=nc, f=f)
        x_visual = self.dropout2(self.project_v(x_visual))
        class_cls = rearrange(self.cls, 'ch -> 1 1 ch').repeat(bs, 1, 1)
        
        
        combined = self.pos(x_visual)
        combined  = torch.cat((class_cls, combined), 1)
        #combined = self.transformer(combined)
        #classification = self.classification(x).squeeze(-1)  # (B, C)

        classification = self.classification(combined[:,0])
        
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