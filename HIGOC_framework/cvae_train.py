from cvae import Encoder, Decoder, CVAE, BaselineNet
import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, TensorDataset
from tqdm import tqdm
from copy import deepcopy
import h5py

class CVAETrainer(nn.Module):
    def __init__(self, data, b, train_test_split_ratio, state_dim, base_model_path):
        super(CVAETrainer, self).__init__()
        with h5py.File(data, 'r') as hdf:
            self.states = torch.tensor(hdf['observations'])
            self.actions = torch.tensor(hdf['actions'])
            self.rewards = torch.tensor(hdf['rewards'])
            self.next_states = torch.tensor(hdf['next_observations'])
        
        self.b = b
        self.train_test_split_ratio = train_test_split_ratio

        states_dataset = TensorDataset(self.states)

        train_size = int(self.train_test_split_ratio * len(states_dataset))
        test_size = len(states_dataset) - train_size

        train_dataset, test_dataset = random_split(states_dataset, [train_size, test_size])

        self.train_loader = DataLoader(train_dataset, batch_size=b, shuffle=True)
        self.test_loader = DataLoader(test_dataset, batch_size=b, shuffle=True)

        self.dataloaders = {'train':self.test_loader, 'val':self.test_loader}

        self.state_dim = state_dim

        self.base_model_path = base_model_path
    
    def train_baseline(self, num_epochs, optimizer, criterion, baseline_model, early_stop_patience = 10, hidden1=128, hidden2=128):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        baseline_model = baseline_model.to(device)
        best_loss = np.inf
        early_stop_count = 0

        for epoch in range(num_epochs):
            for phase in ["train", "val"]:
                if phase == "train":
                    baseline_model.train()
                else:
                    baseline_model.eval()

                running_loss = 0.0
                num_preds = 0

                bar = tqdm(
                    self.dataloaders[phase], desc="NN Epoch {} {}".format(epoch, phase).ljust(20)
                )
                for i, batch in enumerate(bar):
                    x = batch.to(device)

                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == "train"):
                        preds = baseline_model(x)
                        loss = criterion(preds, x) / x.size(0)
                        if phase == "train":
                            loss.backward()
                            optimizer.step()

                    running_loss += loss.item()
                    num_preds += 1
                    if i % 10 == 0:
                        bar.set_postfix(
                            loss="{:.2f}".format(running_loss / num_preds),
                            early_stop_count=early_stop_count,
                        )

                epoch_loss = running_loss / len(self.dataloaders[phase])
                # deep copy the model
                if phase == "val":
                    if epoch_loss < best_loss:
                        best_loss = epoch_loss
                        best_model_wts = deepcopy(baseline_model.state_dict())
                        early_stop_count = 0
                    else:
                        early_stop_count += 1

            if early_stop_count >= early_stop_patience:
                break

        torch.save(baseline_model.state_dict(), f"{self.base_model_path}/baseline.pth")

        return baseline_model


    def train(self, num_epochs, optimizer, baseline_net, early_stop_patience, cvae_net, z_dim = 200, hidden1 = 500, hidden2 = 500, scheduler=None):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        cvae_net = cvae_net.to(device)
    
        best_loss = np.inf
        early_stop_count = 0

        for epoch in range(num_epochs):
            for phase in ["train", "val"]:
                running_loss = 0.0
                num_preds = 0

                # Iterate over data.
                bar = tqdm(
                    self.dataloaders[phase],
                    desc="CVAE Epoch {} {}".format(epoch, phase).ljust(20),
                )
                for i, batch in enumerate(bar):
                    x = batch.to(device)

                    if phase == "train":
                        cvae_net.train()
                        y_mean, y_std, z_mean_rec, z_std_rec, z_mean_prior, z_std_prior = cvae_net(x, x)
                        loss = cvae_net.compute_loss(x, y_mean, y_std, z_mean_rec, z_std_rec, z_mean_prior, z_std_prior)
                        loss.backward()
                        optimizer.step()
                    else:
                        cvae_net.eval()
                        y_mean, y_std, z_mean_rec, z_std_rec, z_mean_prior, z_std_prior = cvae_net(x)
                        loss = cvae_net.compute_loss(x, y_mean, y_std, z_mean_rec, z_std_rec, z_mean_prior, z_std_prior)

                    # statistics
                    running_loss += loss 
                    num_preds += 1
                    if i % 10 == 0:
                        bar.set_postfix(
                            loss="{:.2f}".format(running_loss / num_preds),
                            early_stop_count=early_stop_count,
                        )

                epoch_loss = running_loss / len(self.dataloaders[phase])
                # deep copy the model
                if phase == "val":
                    if epoch_loss < best_loss:
                        best_loss = epoch_loss
                        torch.save(cvae_net.state_dict(), f"{self.base_model_path}/cvae.pth")
                        early_stop_count = 0
                    else:
                        early_stop_count += 1

            if early_stop_count >= early_stop_patience:
                break

        return cvae_net

if __name__ == '__main__':
    file = "D:\IIT Madras\Sem 7\Advances in RL\DA7400-Project\HIGOC_framework\Ant_maze_u-maze_noisy_multistart_False_multigoal_False_sparse.hdf5"
    trainer_obj = CVAETrainer(file, 32, 0.8, 111, "HIGOC_framework/weights")
    baseline_model = BaselineNet(111, 128, 128)
    optimizer = torch.optim.Adam(baseline_model.parameters(), 1e-3)
    criterion = nn.MSELoss()
    baseline_net = trainer_obj.train_baseline(100, optimizer, criterion, baseline_model)

    cvae_model = CVAE(111, 200, 500, 500, baseline_net)
    cvae_model = trainer_obj.train(100, optimizer, baseline_net, 10, cvae_model)