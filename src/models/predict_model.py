import torch
from argparse import ArgumentParser
from torch.utils.data import TensorDataset, DataLoader
from torch import optim
from autoencoder import Autoencoder
import pandas as pd
from sklearn.manifold import TSNE

def _setup_parser():
    parser = ArgumentParser(add_help=True)
    parser.add_argument('--modelpath', default="data/output/model_save.pth", type=str)
    parser.add_argument('--newdatapath', default="/storage", type=str)
    # parser.add_argument('--pathoutput', default="data/output", type=str)
    # parser.add_argument('--device', default="cpu", type=str)
    # parser.add_argument('--batch_size', default=512, type=int)
    # parser.add_argument('--learning_rate', default=1e-3, type=float)
    # parser.add_argument('--epochs', default=1500, type=int)
    # parser.add_argument('--H', default=200, type=int)
    # parser.add_argument('--H2', default=50, type=int)
    # parser.add_argument('--H3', default=12, type=int)
    # parser.add_argument('--latentdims', default=5, type=int)
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

def main():
    parser = _setup_parser()
    args = parser.parse_args()
    model, optimizer = _load_model(args.modelpath)
    newdata = _read_new_data(args.newdatapath)
    mu_result, _ = _predict(model, optimizer, newdata)
    tsne_df_new = _clustering(mu_result)
    # tsne_df.to_parquet(
        # (path_latents / "tsne_df.parquet").resolve()
    # )
    return tsne_df_new

if __name__ == "__main__":
    out = main()
    print(out)
