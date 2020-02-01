import torch
from torch import nn
from torch.nn import functional as F

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class UnFlatten(nn.Module):
    def forward(self, input, size=256):
        return input.view(input.size(0), size, 5, 8)


class VAE(nn.Module):
    def __init__(self, image_channels=3, h_dim=10240, z_dim=32):
        super(VAE, self).__init__()
        self.z_dim = z_dim
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.encoder = nn.Sequential(
            nn.Conv2d(image_channels, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2),
            nn.ReLU(),
            Flatten()
        )

        self.fc1 = nn.Linear(h_dim, z_dim)
        self.fc2 = nn.Linear(h_dim, z_dim)
        self.fc3 = nn.Linear(z_dim, h_dim)

        self.decoder = nn.Sequential(
            UnFlatten(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=(5,4), stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=(5,4), stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, image_channels, kernel_size=(8,6), stride=2),
            nn.Sigmoid(),
        )

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        # return torch.normal(mu, std)
        esp = torch.randn(*mu.size()).to(self.device)
        z = mu + std * esp
        return z

    def bottleneck(self, h):
        mu, logvar = self.fc1(h), F.softplus(self.fc2(h))
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def encode(self, x):
        h = self.encoder(x)
        z, mu, logvar = self.bottleneck(h)
        return z, mu, logvar

    def decode(self, z):
        z = self.fc3(z)
        z = self.decoder(z)
        return z

    def forward(self, x):
        z, mu, logvar = self.encode(x)
        z = self.decode(z)
        return z, mu, logvar

    def loss_fn(self, x):
        z, mean, var = self.encode(x)
        KL = -0.5 * torch.mean(torch.sum(1 + torch.log(var) - mean**2 - var))
        #kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        y = self.decode(z)
        reconstruction = F.binary_cross_entropy(y.view(-1,57600), x.view(-1, 57600), size_average=False)
        #reconstruction = torch.mean(torch.sum(x * torch.log(y) + (1 - x) * torch.log(1 - y)))
        #lower_bound = [-KL, reconstruction]
        return KL+reconstruction#-sum(lower_bound)