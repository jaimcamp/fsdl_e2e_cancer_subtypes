import torch
from argparse import ArgumentParser
from torch.utils.data import TensorDataset, DataLoader
from torch import optim
from autoencoder import Autoencoder
import pandas as pd
from sklearn.manifold import TSNE
from pathlib import Path

def _setup_parser():
    parser = ArgumentParser(add_help=True)
    parser.add_argument('--modelpath', default="data/output/model_save.pth", type=str)
    parser.add_argument('--newdatapath', default="/storage/newdata.csv", type=str)
    parser.add_argument('--originaldatapath', default="/storage/tsn_df.csv", type=str)
    return parser

def _read_new_data(path):
    dataset = torch.tensor(
        pd.read_csv(path).values
    )
    dataset = TensorDataset(dataset.float())
    new_batches = DataLoader(dataset, batch_size=8, shuffle=True)
    return new_batches

def _load_model(model_path):
    checkpoint = torch.load(model_path)
    model = Autoencoder(19875, 200, 50, 12, 5)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return model, optimizer

def _predict(model, optimizer, newdata):
    mu_output = []
    logvar_output = []
    with torch.no_grad():
        for i, data in enumerate(newdata):
            data = data[0]
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(data)
            mu_tensor = mu
            mu_output.append(mu_tensor)
            mu_result = torch.cat(mu_output, dim=0)
            logvar_tensor = logvar
            logvar_output.append(logvar_tensor)
            logvar_result = torch.cat(logvar_output, dim=0)
    return mu_result, logvar_result

def _clustering(latent_vars):
    tsne = TSNE(n_components=2, verbose=0, perplexity=40, n_iter=300)
    tsne_vae_results = tsne.fit_transform(latent_vars)
    tsne_df = pd.DataFrame(tsne_vae_results,
                           columns=["First Dimension", "Second Dimension"]
                           )
    tsne_df['Project'] = 'Uploaded Data'
    return tsne_df

def _read_old_tse(path):
    return pd.read_parquet(path)

def main():
    parser = _setup_parser()
    args = parser.parse_args()
    model, optimizer = _load_model(args.modelpath)
    newdata = _read_new_data(args.newdatapath)
    mu_result, _ = _predict(model, optimizer, newdata)
    tsne_df_new = _clustering(mu_result)
    tsne_df_old = _read_old_tse(args.originaldatapath)
    tsne_df_cat = pd.concat([tsne_df_new, tsne_df_old])
    oldpath = Path(args.originaldatapath)
    breakpoint()
    tsne_df_old.to_parquet(
        oldpath.parent / (oldpath.stem + '.old' + oldpath.suffix)
    )
    tsne_df_cat.to_parquet(oldpath)
    return tsne_df_cat

if __name__ == "__main__":
    out = main()
    print(out)
