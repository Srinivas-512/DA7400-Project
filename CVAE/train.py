import os
import gym
import d4rl
import gym.envs
import torch
from cvae import *
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

current_folder = os.path.dirname(os.path.abspath(__file__))
weights_folder = os.path.join(current_folder, "weights")

if not os.path.exists(weights_folder):
    os.makedirs(weights_folder)


class Trainer(nn.Module):
    '''
    Class to train CVAE for high level planner
    '''

    def __init__(self, env_name:str, subgoal_patience:int):
        '''
        Params : subgoal_patience (int) -> Expected timesteps to reach subgoal
        '''
        super(Trainer, self).__init__()

        env = gym.make(env_name)
        self.env_name = env_name
        dataset = env.get_dataset()
        self.state_dim = env.observation_space.shape[0]

        self.subgoal_patience = subgoal_patience

        self.pairs = self._get_data(dataset)
    
    def _get_data(self, dataset):
        '''
        Function to organize dataset into pairs of training examples for CVAE
        '''

        observations = dataset["observations"]
        terminals = dataset["terminals"]
        timeouts = dataset["timeouts"]

        trajectories = []
        current_trajectory = []
        for i in range(len(observations)):
            current_trajectory.append(observations[i])
            
            if terminals[i] or timeouts[i]:
                trajectories.append(current_trajectory)
                current_trajectory = []

        pairs = []
        for trajectory in trajectories:
            trajectory_length = len(trajectory)
            for t in range(trajectory_length - self.subgoal_patience):
                s_t = trajectory[t]
                s_t[-(self.state_dim-2):] = 0.0
                s_t_N = trajectory[t + self.subgoal_patience]
                s_t_N[-(self.state_dim-2):] = 0.0
                pairs.append((s_t, s_t_N))

        pairs = torch.tensor(np.array(pairs))
        print("Total number of pairs:", len(pairs))
        print("Example pair:", pairs[0])

        return pairs
    
    def _loss_function(self, reconstructed, gti_next, mu, logvar):
        '''
        Find CVAE loss
        '''
        recon_loss = F.mse_loss(reconstructed, gti_next)
        kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return recon_loss + kl_divergence, recon_loss , kl_divergence
    
    def train(
            self, lr : float, epochs : int, batch_size : int, device : str, 
            latent_dim : int, hidden_size : int, use_scheduler : bool = True):
        
        '''
        Main trainer function
        Params : 
            lr -> learning rate
            epochs -> number of epochs to train
            batch_size -> batch size for training
            device -> device (cuda or cpu)
            latent_dim -> dimension of latent vectors in CVAE (8 for antmaze)
            hidden_size -> hidden layer size in CVAE
            use_scheduler -> whether or not to use cosine annealing lr scheduling
        '''

        model = CVAE(self.state_dim, latent_dim, hidden_size).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)

        if use_scheduler:
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs) 

        for epoch in range(epochs):
            model.train()
            perm = torch.randperm(self.pairs.size(0))  # Shuffle the dataset
            loss_avg = 0
            recon_avg = 0
            kl_avg = 0

            for i in range(0, self.pairs.size(0), batch_size):
                optimizer.zero_grad()

                indices = perm[i:i + batch_size]
                batch_current_goals = self.pairs[indices, 0].to(device)
                batch_next_goals = self.pairs[indices, 1].to(device)

                gti_next_reconstructed, mu, logvar = model(batch_current_goals, batch_next_goals)

                loss, recon_loss, kl_divergence = self._loss_function(gti_next_reconstructed, batch_next_goals, mu, logvar)
                loss_avg += loss.item()
                recon_avg += recon_loss.item()
                kl_avg += kl_divergence.item()

                loss.backward()
                # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            if use_scheduler:            
                scheduler.step()

            print(f'Epoch {epoch + 1}, Loss: {loss_avg/(self.pairs.size(0) / batch_size)}, Recon loss : {recon_avg/(self.pairs.size(0) / batch_size)}, KL loss : {kl_avg/(self.pairs.size(0) / batch_size)}')
        
        output_file_path = os.path.join(weights_folder, f"{self.env_name}_{latent_dim}_{hidden_size}.pth")        
        torch.save(model.state_dict(), output_file_path)

if __name__ == '__main__':
    '''
    Numbers here an example, not necessarily indicative of values used in experimentation
    '''
    obj = Trainer("antmaze-umaze-diverse-v0", 10)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    obj.train(lr = 0.01, epochs = 3, batch_size = 2048, device = device, latent_dim = 8, hidden_size = 64, use_scheduler = True)