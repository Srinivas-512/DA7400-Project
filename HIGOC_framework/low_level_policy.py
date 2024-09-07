import torch
import torch.nn as nn
import torch.optim as optim

class LowLevelPolicy(nn.Module):
    def __init__(self, state_dim, action_dim, goal_dim, hidden_dim):
        super(LowLevelPolicy, self).__init__()
        self.fc1 = nn.Linear(state_dim + goal_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, state, goal):
        x = torch.cat([state, goal], dim=-1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        action = self.fc3(x)
        return action

class LowLevelAgent:
    def __init__(self, state_dim, action_dim, goal_dim, hidden_dim, lr=1e-3):
        self.policy = LowLevelPolicy(state_dim, action_dim, goal_dim, hidden_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

    def select_action(self, state, goal):
        with torch.no_grad():
            return self.policy(state, goal)

    def update(self, state, goal, action, loss_fn):
        predicted_action = self.policy(state, goal)
        loss = loss_fn(predicted_action, action)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

# Example usage
if __name__ == "__main__":
    state_dim = 32
    action_dim = 8
    goal_dim = 16
    hidden_dim = 64

    agent = LowLevelAgent(state_dim, action_dim, goal_dim, hidden_dim)
    state = torch.randn(16, state_dim)
    goal = torch.randn(16, goal_dim)
    action = torch.randn(16, action_dim)

    loss_fn = nn.MSELoss()
    agent.update(state, goal, action, loss_fn)
