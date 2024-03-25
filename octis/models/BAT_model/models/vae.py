#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   vae.py
@Time    :   2020/09/30 15:07:10
@Author  :   Leilan Zhang
@Version :   1.0
@Contact :   zhangleilan@gmail.com
@Desc    :   None
'''


import torch
import torch.nn as nn
import torch.nn.functional as F

# VAE model
class VAE(nn.Module):
    def __init__(self, encode_dims=[2000,1024,512,20],decode_dims=[20,1024,2000],dropout=0.0, neg_dec=False, topK=None):

        super(VAE, self).__init__()
        self.encoder = nn.ModuleDict({
            f'enc_{i}':nn.Linear(encode_dims[i],encode_dims[i+1]) 
            for i in range(len(encode_dims)-2)
        })
        self.fc_mu = nn.Linear(encode_dims[-2],encode_dims[-1])
        self.fc_logvar = nn.Linear(encode_dims[-2],encode_dims[-1])

        self.decoder = nn.ModuleDict({
            f'dec_{i}':nn.Linear(decode_dims[i],decode_dims[i+1])
            for i in range(len(decode_dims)-1)
        })
        self.latent_dim = encode_dims[-1]
        self.dropout = nn.Dropout(p=dropout)
        self.fc1 = nn.Linear(encode_dims[-1],encode_dims[-1])
        
        self.neg_dec=neg_dec
        self.topK=topK
        
    def encode(self, x):
        hid = x
        for i,layer in self.encoder.items():
            hid = F.relu(self.dropout(layer(hid)))
        mu, log_var = self.fc_mu(hid), self.fc_logvar(hid)
        return mu, log_var

    def inference(self,x):
        mu, log_var = self.encode(x)
        theta = torch.softmax(x,dim=1)
        return theta
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        hid = z
        for i,(_,layer) in enumerate(self.decoder.items()):
            hid = layer(hid)
            if i<len(self.decoder)-1:
                hid = F.relu(self.dropout(hid))
        return hid
    
    def forward(self, x, collate_fn=None):
        mu, log_var = self.encode(x)
        _theta = self.reparameterize(mu, log_var)
        _theta = self.fc1(_theta) 
        if collate_fn!=None:
            theta = collate_fn(_theta)
        else:
            theta = _theta

        x_reconst = self.decode(theta)
        #For -ve sampling on decoder
        if self.neg_dec:
            theta_neg = self.perturbTheta(theta, self.topK)
            x_reconst_neg = self.decode(theta_neg)
            return x_reconst, mu, log_var, x_reconst_neg
        else:
            return x_reconst, mu, log_var
    
    @staticmethod
    def perturbTopK(x, k):
        _, kidx = x.topk(k=k, dim=1)
        y = x.clone()
        y[torch.arange(y.size(0))[:, None], kidx] = 0.0
        return y

    @staticmethod
    def perturbTheta(x, k):
        x_new =  VAE.perturbTopK(x, k)
        x_new = x_new/x_new.sum(dim=-1).unsqueeze(1)
        return x_new

if __name__ == '__main__':
    model = VAE(encode_dims=[1024,512,256,20],decode_dims=[20,128,768,1024])
    model = model.cuda()
    inpt = torch.randn(234,1024).cuda()
    out,mu,log_var = model(inpt)
    print(out.shape)
    print(mu.shape)