---
author: Jaime Campos
title: Proposal for FSDL project
date: Wed Mar 31 04:27:31 PM CEST 2021
---

# FSLS 2021 project - GenoType Explorer, an E2E classification an exploration of tumor's subtypes

-----
__DESCRIPTION__
This is a project for the Full Stack Deep Learning course from 2021.
The main objective is to build an end-to-end Webapp for the clustering and classification of cancer samples.
This project mixes an academic approach to analyze and classify samples with an end-user interface to study the subtypes and compare to their samples.

-----

## Outcome

The final outcome of this project is a Webapp, where a user can upload their transcriptomic data for their sample, and they  will be classify in one of the previous discovered subtypes.

In addition, information about the distance to the other subtypes and general information about them will also be provided.

## Methodology

This project is divided into 2 main parts:

### Clustering of samples of brain tumor

- [x] Freely available transcriptomic data is obtained from public repositories
- [x] Set up the use of PyTorch with AMD GPUs
- [x] Implement a Variational Autoencoder to use with the transcriptomic tabular data
- [x] Perform clustering of samples using latent variables
- [ ] Debug implementation of VAE

### Webapp for upload and analysis of samples

- [x] Implement Streamlit webapp to explore found clusters
- [x] Add upload and classification capacity to webapp
- [x] Set up backend with Fastapi to calculate latent variables for uploaded samples
- [ ] Subtype analysis of the samples
- [ ] Profiling of the groups of samples

## Data type and sources

At the beginning only transcriptomic data (RNAseq), if possible add additional types of data such as microRNA, clinical and other.

### Sources

- The Cancer Genome Atlas Program - TCGA: Over 20,000 samples of cancer with genomic, epigenomic, transcriptomic, and proteomic data.

