#! /usr/bin/env python

#%%
import pandas as pd
import torch

#%%
data = pd.read_parquet("data/processed/complete_dataset.parquet")
torch.save(torch.tensor(data.values), "data/processed/complete_dataset.pt")

#%%
mini = data.iloc[:5000]
del data
torch.save(torch.tensor(mini.values), "data/processed/mini_dataset.pt")
mini.to_parquet("data/processed/mini_dataset.parquet")

#%%
import json
with open('data/'):
