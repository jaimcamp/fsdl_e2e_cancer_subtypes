# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---


# %%
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from torch.nn import functional as F
from sklearn.cluster import KMeans, DBSCAN
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from torch.autograd import Variable

# %% tags=[]

import wandb

# 1. Start a new run
wandb.init(project='gt-explorer', entity='jaimcamp')

# 2. Save model inputs and hyperparameters
config = wandb.config
config.learning_rate = 0.0001
config.epochs = 1500
config.H = 200
config.H2 = 50
config.H3 = 12
config.final = 5


# %%
# Load Data

# %%
torch.cuda.device_count()

# %%
del data_X, data_y, data_set

# %%
device = torch.device('cuda')
# data_X = torch.load("data/processed/mini_dataset.pt", device)
data_y = torch.load("data/processed/data_y.pt", device)
data_set = TensorDataset(torch.load("data/processed/selected_data.pt", device).float())
#                          torch.load("data/processed/mini_data_y.pt", device).long())

# %%
train_batches = DataLoader(data_set, batch_size=1024, shuffle=True)

# %%
# device = torch.device('cuda')
# data_X = torch.load("data/processed/mini_dataset.pt", device)
# data_y = torch.load("data/processed/mini_data_y.pt", device)
# data_set = TensorDataset(data_X.float(), data_y.long())

# %%
from pl_bolts.models.self_supervised import SimCLR
from pl_bolts.models.self_supervised.simclr.transforms import SimCLRTrainDataTransform, SimCLREvalDataTransform
import pytorch_lightning as pl
from pl_bolts.models.autoencoders import AE


# %%
class Autoencoder(nn.Module):
    def __init__(self,D_in,H=50,H2=12, H3=5,latent_dim=3):
        
        #Encoder
        super(Autoencoder,self).__init__()
        self.linear1=nn.Linear(D_in,H)
        self.lin_bn1 = nn.BatchNorm1d(num_features=H)
        self.linear2=nn.Linear(H,H2)
        self.lin_bn2 = nn.BatchNorm1d(num_features=H2)
        self.linear3=nn.Linear(H2,H3)
        self.lin_bn3 = nn.BatchNorm1d(num_features=H3)
        self.linear4=nn.Linear(H3,H3)
        self.lin_bn4 = nn.BatchNorm1d(num_features=H3)
        
#         # Latent vectors mu and sigma
        self.fc1 = nn.Linear(H3, latent_dim)
        self.bn1 = nn.BatchNorm1d(num_features=latent_dim)
        self.fc21 = nn.Linear(latent_dim, latent_dim)
        self.fc22 = nn.Linear(latent_dim, latent_dim)

#         # Sampling vector
        self.fc3 = nn.Linear(latent_dim, latent_dim)
        self.fc_bn3 = nn.BatchNorm1d(latent_dim)
        self.fc4 = nn.Linear(latent_dim, H3)
        self.fc_bn4 = nn.BatchNorm1d(H3)
        
#         # Decoder
        self.linear4a=nn.Linear(H3,H3)
        self.lin_bn4a = nn.BatchNorm1d(num_features=H3)
        self.linear4b=nn.Linear(H3,H2)
        self.lin_bn4b = nn.BatchNorm1d(num_features=H2)
        self.linear5=nn.Linear(H2,H)
        self.lin_bn5 = nn.BatchNorm1d(num_features=H)
        self.linear6=nn.Linear(H,D_in)
        self.lin_bn6 = nn.BatchNorm1d(num_features=D_in)
        
        self.relu = nn.ReLU()
        
    def encode(self, x):
        lin1 = self.relu(self.lin_bn1(self.linear1(x)))
        lin2 = self.relu(self.lin_bn2(self.linear2(lin1)))
        lin3 = self.relu(self.lin_bn3(self.linear3(lin2)))
        lin4 = self.relu(self.lin_bn4(self.linear4(lin3)))
        
        fc1 = F.relu(self.bn1(self.fc1(lin4)))

        r1 = self.fc21(fc1)
        r2 = self.fc22(fc1)
        
        return r1, r2
    
    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu
        
    def decode(self, z):
        fc3 = self.relu(self.fc_bn3(self.fc3(z)))
        fc4 = self.relu(self.fc_bn4(self.fc4(fc3)))

        lin4a = self.relu(self.lin_bn4a(self.linear4a(fc4)))
        lin4b = self.relu(self.lin_bn4b(self.linear4b(lin4a)))
        lin5 = self.relu(self.lin_bn5(self.linear5(lin4b)))
        return self.lin_bn6(self.linear6(lin5))


        
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        # self.decode(z) ist später recon_batch, mu ist mu und logvar ist logvar
        return self.decode(z), mu, logvar


# %%
class customLoss(nn.Module):
    def __init__(self):
        super(customLoss, self).__init__()
        self.mse_loss = nn.MSELoss(reduction="sum")
    
    # x_recon ist der im forward im Model erstellte recon_batch, x ist der originale x Batch, mu ist mu und logvar ist logvar 
    def forward(self, x_recon, x, mu, logvar):
        loss_MSE = self.mse_loss(x_recon, x)
        loss_KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return loss_MSE + loss_KLD



# %%
# takes in a module and applies the specified weight initialization
def weights_init_uniform_rule(m):
    classname = m.__class__.__name__
    # for every Linear layer in a model..
    if classname.find('Linear') != -1:
        # get the number of the inputs
        n = m.in_features
        y = 1.0/np.sqrt(n)
        m.weight.data.uniform_(-y, y)
        m.bias.data.fill_(0)

# %%


D_in = data_set.tensors[0].shape[1]
H = config.H
H2 = config.H2
H3 = config.H3
final = config.final
model = Autoencoder(D_in, H, H2, H3, final).to(device)
model.apply(weights_init_uniform_rule)
optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)



# %%


loss_mse = customLoss()



# %%


epochs = config.epochs
log_interval = 5
val_losses = []
train_losses = []
wandb.watch(model)


# %%
def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, data in enumerate(train_batches):
#         breakpoint()
        data = data[0].to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_mse(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % log_interval == 0:
            wandb.log({"loss": loss})
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
               epoch, batch_idx * len(data), len(train_batches.dataset),
                      100. * batch_idx / len(train_batches),
                      loss.item() / len(data)))
    if epoch % 200 == 0:        
        print('====> Epoch: {} Average loss: {:.4f}'.format(
            epoch, train_loss / len(train_batches.dataset)))
        train_losses.append(train_loss / len(train_batches.dataset))


# %%


for epoch in range(1, epochs + 1):
    train(epoch)



# %%
train_losses

# %%
