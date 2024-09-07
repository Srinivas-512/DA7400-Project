import torch
import torch.nn as nn


class BaselineNet(nn.Module):
    def __init__(self, state_dim, hidden_1, hidden_2):
        super().__init__()
        self.state_dim = state_dim
        self.fc1 = nn.Linear(state_dim, hidden_1)
        self.fc2 = nn.Linear(hidden_1, hidden_2)
        self.fc31 = nn.Linear(hidden_2, state_dim)
        self.fc32 = nn.Linear(hidden_2, state_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(-1, self.state_dim)
        hidden = self.relu(self.fc1(x))
        hidden = self.relu(self.fc2(hidden))
        # y = torch.sigmoid(self.fc3(hidden))
        y_mean = self.fc31(hidden)
        y_std = torch.exp(self.fc32(hidden))
        return y_mean, y_std


class Encoder(nn.Module):
    def __init__(self, state_dim, z_dim, hidden_1, hidden_2):
        super().__init__()
        self.state_dim = state_dim
        self.fc1 = nn.Linear(state_dim, hidden_1)
        self.fc2 = nn.Linear(hidden_1, hidden_2)
        self.fc31 = nn.Linear(hidden_2, z_dim)
        self.fc32 = nn.Linear(hidden_2, z_dim)
        self.relu = nn.ReLU()

    def forward(self, x, y):
        xc = x.clone()
        xc[x == -1] = y[x == -1]
        xc = xc.view(-1, self.state_dim)
        hidden = self.relu(self.fc1(xc))
        hidden = self.relu(self.fc2(hidden))
        z_mean = self.fc31(hidden)
        z_std = torch.exp(self.fc32(hidden))
        return z_mean, z_std


class Decoder(nn.Module):
    def __init__(self, state_dim, z_dim, hidden_1, hidden_2):
        super().__init__()
        self.state_dim = state_dim
        self.fc1 = nn.Linear(z_dim, hidden_1)
        self.fc2 = nn.Linear(hidden_1, hidden_2)
        self.fc31 = nn.Linear(hidden_2, state_dim)
        self.fc32 = nn.Linear(hidden_2, state_dim)
        self.relu = nn.ReLU()

    def forward(self, z):
        y = self.relu(self.fc1(z))
        y = self.relu(self.fc2(y))
        # y = torch.sigmoid(self.fc3(y))
        y_mean = self.fc31(y)
        y_std = torch.exp(self.fc32(y))
        return y_mean, y_std


class CVAE(nn.Module):
    def __init__(self, state_dim, z_dim, hidden_1, hidden_2, pre_trained_baseline_net):
        super().__init__()
        self.state_dim = state_dim
        self.baseline_net = pre_trained_baseline_net
        self.prior_net = Encoder(state_dim, z_dim, hidden_1, hidden_2)
        self.generation_net = Decoder(state_dim, z_dim, hidden_1, hidden_2)
        self.recognition_net = Encoder(state_dim, z_dim, hidden_1, hidden_2)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, xs, ys=None):
        y_hat_mean, y_hat_std = self.baseline_net(xs)

        y_hat = torch.normal(y_hat_mean, y_hat_std).view(xs.shape)

        z_mean_rec, z_std_rec = self.recognition_net(xs, ys)

        z_mean_prior, z_std_prior = self.prior_net(xs, y_hat)

        # Use recognition network during training, prior network during inference
        if ys is not None:
            zs = self.reparameterize(z_mean_rec, torch.log(z_std_rec))
        else:
            zs = self.reparameterize(z_mean_prior, torch.log(z_std_prior))

        # Generate output y
        y_mean, y_std = self.generation_net(zs)
        return y_mean, y_std, z_mean_rec, z_std_rec, z_mean_prior, z_std_prior

    def compute_loss(self, xs, y_mean, y_std, z_mean_rec, z_std_rec, z_mean_prior, z_std_prior):
        recon_loss = torch.sum(-0.5*torch.log(2*torch.pi*torch.ones_like(y_std)) - torch.log(y_std) - (xs-y_mean)**2/(2*y_std**2))
        kl_div = -torch.sum(-0.5*torch.ones_like(z_mean_rec) + torch.log(z_std_prior) - torch.log(z_std_rec) + (z_std_rec**2 + (z_mean_rec-z_mean_prior)**2)/(2*z_std_prior**2))
        
        return torch.mean(-recon_loss - kl_div)

    def save(self, model_path):
        torch.save(self.state_dict(), model_path)

    def load(self, model_path, map_location=None):
        self.load_state_dict(torch.load(model_path, map_location=map_location))
        self.eval()


if __name__ == '__main__':
    baseline_net = BaselineNet(111, 128, 64)
    cvae_net = CVAE(111, 200, 500, 500, baseline_net)
    batch_size = 16
    xs = torch.randn(batch_size, 111)
    # ys = torch.randint(0, 2, (batch_size, 1, 28, 28)).float()
    y_mean, y_std, z_mean_rec, z_std_rec, z_mean_prior, z_std_prior = cvae_net(xs, xs)
    print(cvae_net.compute_loss(xs, y_mean, y_std, z_mean_rec, z_std_rec, z_mean_prior, z_std_prior))