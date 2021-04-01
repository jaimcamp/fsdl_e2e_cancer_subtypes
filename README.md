# FSLS 2021 proeject - E2E classification of brain tumor's subtypes

-----
This is a project for the Full Stack Deep Learning course from 2021.
The main objective is to build an end-to-end Webapp for the clustering and classification of brain tumor samples.
This project mixes an academic approach to analyse and classify samples, and to give the end-user an interface to study the subtypes and compare to their samples.

-----

## Outcome

The final outcome of this project is a Webapp, where a user can upload their transcriptomic data and their sample will be classify in one of the previous discovered subtypes.

In addition, information about the distance to the other subtypes and general information about them will also be provided.

## Methodology

This project is divided into 2 main parts:

### Clustering of samples of brain tumor

- [ ] Freely available transcriptomic data is obtained from public repositories
- [ ] Perform clustering of samples using an autoencoder architecture
- [ ] Find main attributes of clusters

### Webapp for upload and analysis of samples

- [ ] Implement Webapp to explore found clusters
- [ ] Add upload and classification capacity

## Data type and sources

At the beginning only transcriptomic data (RNAseq), if possible add additional types of data such as microRNA, clinical and other.

### Sources

- The Cancer Genome Atlas Program - TCGA: Over 20,000 samples of cancer with genomic, epigenomic, transcriptomic, and proteomic data.
- The Genotype-Tissue Expression - GTEx: Over 1,000 non-disease samples with gene expression and regulation data.

