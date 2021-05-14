import numpy as np
import pandas as pd
from sklearn.manifold import TSNE


def _read_latent_vars(path_latents):
    out_file = (path_latents / "mu_result.npy").resolve()
    latent_vars = np.load(out_file)
    return latent_vars

def _read_y_class(path_data):
    y_data = (path_data / "projects.npy").resolve()
    return np.load(y_data)

def clustering(path_data, path_latents):
    latent_vars = _read_latent_vars(path_latents)
    y_vars = _read_y_class(path_data)
    tsne = TSNE(n_components=2, verbose=0, perplexity=40, n_iter=300)
    tsne_vae_results = tsne.fit_transform(latent_vars)
    tsne_df = pd.DataFrame(tsne_vae_results,
                           columns=["First Dimension", "Second Dimension"]
                           )
    tsne_df['Project'] = y_vars
    tsne_df.to_parquet(
        (path_latents / "tsne_df.parquet").resolve()
    )
    return tsne_df

