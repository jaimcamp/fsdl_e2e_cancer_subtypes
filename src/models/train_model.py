import torch
from argparse import ArgumentParser
import argparse
from torch.utils.data import TensorDataset, DataLoader
from torch import nn, optim
from pathlib import Path
from autoencoder import Autoencoder, customLoss, weights_init_uniform_rule

def _setup_parser():
    parser = ArgumentParser(add_help=True)
    parser.add_argument('--pathdata', default="data/processed", type=str)
    parser.add_argument('--batch_size', default=512, type=int)
    parser.add_argument('--learning_rate', default=1e-3, type=float)
    parser.add_argument('--epochs', default=1500, type=int)
    parser.add_argument('--H', default=200, type=int)
    parser.add_argument('--H2', default=50, type=int)
    parser.add_argument('--H3', default=12, type=int)
    parser.add_argument('--latentdims', default=5, type=int)
    return parser

def _load_data(device, data_path):
    file_path = (data_path / "selected_data.pt").resolve()
    data_set = TensorDataset(
        torch.load(file_path, device).float())
    return data_set

def train(epoch, train_batches, device, optimizer, loss_mse,
          log_interval, model, train_losses):

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
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_batches.dataset),
                100. * batch_idx / len(train_batches),
                loss.item() / len(data)))
    if epoch % 200 == 0:        
        print('====> Epoch: {} Average loss: {:.4f}'.format(
            epoch, train_loss / len(train_batches.dataset)))
        train_losses.append(train_loss / len(train_batches.dataset))

def main():
    parser = _setup_parser()
    args = parser.parse_args()
    data_path = Path(args.pathdata)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_set = _load_data(device, data_path)
    train_batches = DataLoader(data_set, batch_size=args.batch_size, shuffle=True)
    D_in = data_set.tensors[0].shape[1]
    H = args.H
    H2 = args.H2
    H3 = args.H3
    final = args.latentdims
    model = Autoencoder(D_in, H, H2, H3, final).to(device)
    model.apply(weights_init_uniform_rule)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    loss_mse = customLoss()
    epochs = args.epochs
    log_interval = 5
    val_losses = []
    train_losses = []
    for epoch in range(1, epochs + 1):
        train(epoch, train_batches, device, optimizer, loss_mse,
              log_interval, model, train_losses)

if __name__ == "__main__":
    main()
