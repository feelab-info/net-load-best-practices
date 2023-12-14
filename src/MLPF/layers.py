import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .embending import PosEmbedding, RotaryEmbedding

activations = [nn.ReLU(), nn.SELU(), nn.LeakyReLU(), nn.GELU(), nn.SiLU()]
def scale_dot_attention(k, v, q):
    k = k.unsqueeze(1)
    q = q.unsqueeze(1)
    v = v.unsqueeze(1)
    scale = q.shape[-1] ** 0.5
    unnorm_weights = torch.einsum("bjk,bik->bij", k, q) / scale
    weights = torch.softmax(unnorm_weights, dim=-1)
    rep = torch.einsum("bik,bkj->bij", weights, v)
    return rep.squeeze(1)

def create_linear(in_channels, out_channels, bn=False):
    m = nn.Linear(in_channels,out_channels)
    #nn.init.xavier_normal_(m.weight.data)
    nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        
    if m.bias is not None:
        torch.nn.init.constant_(m.bias, 0)
    if bn:
        bn = nn.BatchNorm1d(out_channels)
        m = nn.Sequential(m, bn)
    return m



def FeedForward(dim, expansion_factor = 2, dropout = 0.0, activation=nn.GELU(), bn=True):
    return nn.Sequential(
        create_linear(dim, dim * expansion_factor, bn),
        activation,
        nn.Dropout(dropout),
        create_linear(dim * expansion_factor, dim, bn),
        nn.Dropout(dropout)
    )
    
    


class MLPBlock(nn.Module):
    def __init__(self, in_size=1, 
                     latent_dim: int = 32,
                    features_start=16, 
                    num_layers=4, 
                 context_size=96,
                 activation=nn.ReLU(),
                 bn=True):
        
        super().__init__()
        self.in_size = in_size*context_size
        self.context_size = context_size
       
        layers = [nn.Sequential(create_linear(self.in_size , features_start, bn=False), activation)]
        feats = features_start
        for i in range(num_layers-1):
            layers.append(nn.Sequential(create_linear(feats, feats*2, bn=bn), activation))
            feats = feats*2
        layers.append(nn.Sequential(create_linear(feats, latent_dim, bn=False), activation))
        self.mlp_network =  nn.ModuleList(layers)
       
        
    def forward(self, x):
        x = x.flatten(1,2)
        for m in self.mlp_network:
            x = m(x)
       
        return x





    
class PastEncoder(nn.Module):

    def __init__(self,  hparams, n_channels: int =1):
        super().__init__()
        self.n_out = len(hparams['targets'])
        self.encoder =MLPBlock(in_size=hparams['emb_size'] if hparams['embed_type']!='None' else n_channels,
                                 latent_dim=hparams['latent_size'],
                                 features_start=hparams['latent_size'], 
                                 num_layers=hparams['depth'], 
                                 context_size=hparams['window_size'], 
                                 activation=activations[hparams['activation']])
       
        
        self.norm = nn.LayerNorm(n_channels)
        self.droput = nn.Dropout(hparams['dropout'])
        self.hparams=hparams
        if hparams['embed_type']=='PosEmb':
            self.emb = PosEmbedding(n_channels, hparams['emb_size'], window_size=hparams['window_size'])
        
        elif hparams['embed_type']=='RotaryEmb':
            self.emb = RotaryEmbedding(hparams['emb_size'])

        
    def forward(self, x):
        x = self.norm(x)
        x = self.emb(x)
        x = self.droput(x)
        x = self.encoder(x)
        return x
    
    

    
class FutureEncoder(nn.Module):

    def __init__(self,  hparams, n_channels: int =1):
        super().__init__()
        self.n_out = len(hparams['targets'])
        self.encoder = MLPBlock(in_size=hparams['emb_size'] if hparams['embed_type']!='None' else n_channels,
                                 latent_dim=hparams['latent_size'],
                                 features_start=hparams['latent_size'], 
                                 num_layers=hparams['depth'], 
                                 context_size=hparams['horizon'], 
                                 activation=activations[hparams['activation']])
       
        self.norm = nn.LayerNorm(n_channels)
        self.droput = nn.Dropout(hparams['dropout'])
        self.hparams=hparams
        if hparams['embed_type']=='PosEmb':
            self.emb = PosEmbedding(n_channels, hparams['emb_size'], window_size=hparams['horizon'])
        
        elif hparams['embed_type']=='RotaryEmb':
            self.emb = RotaryEmbedding(hparams['emb_size'])

    
        
    def forward(self, x):
        x = self.norm(x)
        x = self.emb(x)
        x=self.droput(x)
        x = self.encoder(x)
        return x


