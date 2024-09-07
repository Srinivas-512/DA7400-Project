# Main training loop
import torch 
from low_level_policy import LowLevelAgent
from high_level_planner import HighLevelPlanner
if __name__ == "__main__":
    state_dim = 32
    action_dim = 8
    goal_dim = 16
    hidden_dim = 64

    low_level_agent = LowLevelAgent(state_dim, action_dim, goal_dim, hidden_dim)
    planner = HighLevelPlanner(state_dim + goal_dim, latent_dim=32, hidden_dim=128)

    state = torch.randn(1, state_dim)  # Example initial state
    goals = planner.plan(state)

    for goal in goals:
        action = low_level_agent.select_action(state, goal)
        print(f"Action: {action}")
