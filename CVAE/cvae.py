import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    '''
    Encoder of the CVAE
    '''
    def __init__(self, input_dim:int, latent_dim:int, hidden_dim:int):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input_dim * 2, hidden_dim)  # concatenate gti and gti+1
        self.fc2_mu = nn.Linear(hidden_dim, latent_dim)  # mean of z
        self.fc2_logvar = nn.Linear(hidden_dim, latent_dim)  # log variance of z

    def forward(self, gti, gti_next):
        x = torch.cat([gti, gti_next], dim=-1)  # concatenate the inputs
        h = F.relu(self.fc1(x))
        z_mu = self.fc2_mu(h)
        z_logvar = self.fc2_logvar(h)
        z_logvar = torch.clamp(z_logvar, min=-5, max=5)

        return z_mu, z_logvar


def reparameterize(mu, logvar):
    '''
    Reparametrization Trick
    '''
    std = torch.exp(0.5 * logvar)  # standard deviation
    eps = torch.randn_like(std)  # random normal noise
    return mu + eps * std  # sampled z


class Decoder(nn.Module):
    '''
    Decoder module of the CVAE
    '''
    def __init__(self, input_dim:int, latent_dim:int, hidden_dim:int):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(input_dim + latent_dim, hidden_dim)  # concatenate gti and z
        self.fc2 = nn.Linear(hidden_dim, input_dim)  # output gti+1

    def forward(self, gti, z):
        x = torch.cat([gti, z], dim=-1)  # concatenate gti and z
        h = F.relu(self.fc1(x))
        gti_next_reconstructed = self.fc2(h)
        return gti_next_reconstructed


class CVAE(nn.Module):
    '''
    CVAE for goal planning
    Params:
        - input_dim = dimension of input tensor (subgoal dim + state dim)
        - latent_dim = latent dimension of the CVAE
        - hidden_dim = hidden layer size 
    '''
    def __init__(self, input_dim, latent_dim, hidden_dim):
        super(CVAE, self).__init__()
        self.encoder = Encoder(input_dim, latent_dim, hidden_dim)
        self.decoder = Decoder(input_dim, latent_dim, hidden_dim)

    def forward(self, gti, gti_next):
        # Full forward pass
        z_mu, z_logvar = self.encoder(gti, gti_next)
        z = reparameterize(z_mu, z_logvar)
        gti_next_reconstructed = self.decoder(gti, z)
        return gti_next_reconstructed, z_mu, z_logvar

    def encode(self, gti, gti_next):
        # Get encoder output only
        z_mu, z_logvar = self.encoder(gti, gti_next)
        return z_mu, z_logvar

    def decode(self, gti, z):
        # Decode the latent variable to reconstruct the input
        return self.decoder(gti, z)
    

def sample_noisy_subgoal(cvae, batch_goals, batch_prev_goals, p=0.1, device='cpu'):
    '''
    Function to sample noisy subgoals during the training of low level agent
    '''
    mu, log_var= cvae.encode(batch_prev_goals, batch_goals)
    # Perturb the latent space representation with noise, probability p
    noise = torch.randn_like(mu).to(device)

    perturbation_mask = (torch.rand(mu.shape) < p).float().to(device)

    perturbed_latent = reparameterize(mu + perturbation_mask * noise,log_var)

    goal_states = cvae.decoder(batch_prev_goals,perturbed_latent)
    return goal_states



