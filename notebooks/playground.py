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
import torch
import torchvision as vision
from torch.utils.data import DataLoader, TensorDataset
from sklearn.cluster import DBSCAN
from torch import nn

# %%
device = torch.device('cuda')

# %%
data = torch.load("data/processed/mini_dataset.pt", device)

# %%
data.shape

# %%
data_y = torch.load("data/processed/mini_data_y.pt", device)

# %%
data_set = TensorDataset(data.float(), data_y.long())

# %%
train_batches = DataLoader(data_set, batch_size=500, shuffle=True)

# %%
sum(data.var(dim=0) < 1e-2)


# %%
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
#         self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(60483, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
            nn.ReLU()
        )

    def forward(self, x):
#         x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


# %%
model = NeuralNetwork().to(device)
print(model)

# %%
logits = model(data.float())
pred_probab = nn.Softmax(dim=1)(logits)
y_pred = pred_probab.argmax(1)
print(f"Predicted class: {y_pred}")


# %%

for name, param in model.named_parameters():
    print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")


# %%
def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= size
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


# %%
# Initialize the loss function
loss_fn = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
epochs = 10
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_batches, model, loss_fn, optimizer)
#     test_loop(test_dataloader, model, loss_fn)
print("Done!")

# %%
