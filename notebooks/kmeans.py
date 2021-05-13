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
# %cd ..

# %%
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.cluster import KMeans, DBSCAN
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn import metrics

# %%
# Load Data

# %%
data_X = torch.load("data/processed/mini_dataset.pt").numpy()
data_y = torch.load("data/processed/mini_data_y.pt").numpy()

# %%
# Run Kmeans

# %%
X_std = StandardScaler().fit_transform(data_X)

# %%
np.unique(data_y, return_counts=True)

# %%
inertias = []
sils = []
chs = []
dbs = []
sizes = range(20, 40)
for k in sizes:
    print(f"Loop number {k}")
    k2 = KMeans(random_state=42, n_clusters=k)
    k2.fit(X_std)
    inertias.append(k2.inertia_)
    sils.append(
        metrics.silhouette_score(data_y.reshape(-1,1), k2.labels_)
    )
    chs.append(
        metrics.calinski_harabasz_score(
            data_y.reshape(-1,1), k2.labels_
        )
    )
    dbs.append(
        metrics.davies_bouldin_score(
            data_y.reshape(-1,1), k2.labels_
        )
    )
fig, ax = plt.subplots(figsize=(6, 4))
(
    pd.DataFrame(
        {
            "inertia": inertias,
            "silhouette": sils,
            "calinski": chs,
            "davis": dbs,
            "k": sizes,
        }
    )
    .set_index("k")
    .plot(ax=ax, subplots=True, layout=(2, 2))
)

# %%
np.__config__.show()

# %%
