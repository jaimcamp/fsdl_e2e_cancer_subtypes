from torch import nn
import torch
from torch.nn import functional as F
from torch.autograd import Variable
import numpy as np

#######################################################################################
#  From https://www.kaggle.com/schmiddey/variational-autoencoder-with-pytorch-vs-pca  #
#######################################################################################


class Autoencoder(nn.Module):
    def __init__(self,D_in,H=50,H2=12, H3=5,latent_dim=3):

        #Encoder
        super(Autoencoder,self).__init__()
        self.linear1=nn.Linear(D_in,H)
        self.lin_bn1 = nn.BatchNorm1d(num_features=H)
        self.linear2=nn.Linear(H,H2)
        self.lin_bn2 = nn.BatchNorm1d(num_features=H2)
        self.linear3=nn.Linear(H2,H3)
        self.lin_bn3 = nn.BatchNorm1d(num_features=H3)
        self.linear4=nn.Linear(H3,H3)
        self.lin_bn4 = nn.BatchNorm1d(num_features=H3)

#         # Latent vectors mu and sigma
        self.fc1 = nn.Linear(H3, latent_dim)
        self.bn1 = nn.BatchNorm1d(num_features=latent_dim)
        self.fc21 = nn.Linear(latent_dim, latent_dim)
        self.fc22 = nn.Linear(latent_dim, latent_dim)

#         # Sampling vector
        self.fc3 = nn.Linear(latent_dim, latent_dim)
        self.fc_bn3 = nn.BatchNorm1d(latent_dim)
        self.fc4 = nn.Linear(latent_dim, H3)
        self.fc_bn4 = nn.BatchNorm1d(H3)

#         # Decoder
        self.linear4a=nn.Linear(H3,H3)
        self.lin_bn4a = nn.BatchNorm1d(num_features=H3)
        self.linear4b=nn.Linear(H3,H2)
        self.lin_bn4b = nn.BatchNorm1d(num_features=H2)
        self.linear5=nn.Linear(H2,H)
        self.lin_bn5 = nn.BatchNorm1d(num_features=H)
        self.linear6=nn.Linear(H,D_in)
        self.lin_bn6 = nn.BatchNorm1d(num_features=D_in)

        self.relu = nn.ReLU()

    def encode(self, x):
        lin1 = self.relu(self.lin_bn1(self.linear1(x)))
        lin2 = self.relu(self.lin_bn2(self.linear2(lin1)))
        lin3 = self.relu(self.lin_bn3(self.linear3(lin2)))
        lin4 = self.relu(self.lin_bn4(self.linear4(lin3)))

        fc1 = F.relu(self.bn1(self.fc1(lin4)))

        r1 = self.fc21(fc1)
        r2 = self.fc22(fc1)

        return r1, r2

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        fc3 = self.relu(self.fc_bn3(self.fc3(z)))
        fc4 = self.relu(self.fc_bn4(self.fc4(fc3)))

        lin4a = self.relu(self.lin_bn4a(self.linear4a(fc4)))
        lin4b = self.relu(self.lin_bn4b(self.linear4b(lin4a)))
        lin5 = self.relu(self.lin_bn5(self.linear5(lin4b)))
        return self.lin_bn6(self.linear6(lin5))



    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        # self.decode(z) ist später recon_batch, mu ist mu und logvar ist logvar
        return self.decode(z), mu, logvar


class customLoss(nn.Module):
    def __init__(self):
        super(customLoss, self).__init__()
        self.mse_loss = nn.MSELoss(reduction="sum")

    def forward(self, x_recon, x, mu, logvar):
        loss_MSE = self.mse_loss(x_recon, x)
        loss_KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return loss_MSE + loss_KLD

def weights_init_uniform_rule(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        n = m.in_features
        y = 1.0/np.sqrt(n)
        m.weight.data.uniform_(-y, y)
        m.bias.data.fill_(0)
