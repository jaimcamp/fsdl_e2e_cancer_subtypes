import torch
from argparse import ArgumentParser
from torch.utils.data import TensorDataset, DataLoader
from torch import optim
from pathlib import Path
from autoencoder import Autoencoder, customLoss, weights_init_uniform_rule
from build_cluster import clustering
import numpy as np
import pandas as pd

def _read_new_data(path):
    dataset = pd.read_
def predict():
    with torch.no_grad():
        for i, data in enumerate(train_batches):
            data = data[0].to(device)
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(data)
            mu_tensor = mu
            mu_output.append(mu_tensor)
            mu_result = torch.cat(mu_output, dim=0)
            logvar_tensor = logvar
            logvar_output.append(logvar_tensor)
            logvar_result = torch.cat(logvar_output, dim=0)
