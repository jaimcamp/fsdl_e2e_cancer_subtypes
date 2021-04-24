---
author: Jaime Campos
title: Proposal for FSDL project
date: Wed, Mar 31 2021
---

# About me
- From Chile 🇨🇱 living in Germany 🇩🇪
- Bioinformatics & Data Science
- Interest on advantages of Deep Learning
- Focus on creating data products

# Objective
- Use unsupervised learning to find the subtypes of brain cancer (GBM) using transcriptomic data
- Build a Webapp to:
  - Showcase the findings of the clustering
  - Users  can upload their transcriptomic data to classify their data into one subtype

# How to do it

### Clustering of samples of brain tumor

- Freely available transcriptomic data is obtained from public repositories
- Perform clustering of samples using an autoencoder architecture
- Find main attributes of clusters

------------

### Webapp for upload and analysis of samples

- Implement Webapp to explore found clusters
- Add upload and classification capacity

# Data type and sources

Only transcriptomic data (RNAseq)
If possible add additional data, ie microRNA, clinical and other.

- TCGA: Over 20,000 samples of cancer
- GTEx: Over 1,000 non-disease samples

