import os
from typing import Optional
from dataclasses import dataclass

current_folder = os.path.dirname(os.path.abspath(__file__))
weights_folder = os.path.join(current_folder, "weights")

if not os.path.exists(weights_folder):
    os.makedirs(weights_folder)

class Config:
    def __init__(self, agent_name, exp_name):
        self.agent_name = agent_name
        self.exp_name = exp_name
        self.config = self._get_config()
    
    def _get_config(self):
        if self.agent_name == 'iql':
            # IQLTrainConfig.__post_init__()
            IQLTrainConfig.exp_name = self.exp_name
            name = f"{'iql'}-{self.exp_name}"
            checkpoints_path = os.path.join(weights_folder, name)

            if not os.path.exists(checkpoints_path):
                os.makedirs(checkpoints_path)

            return IQLTrainConfig, checkpoints_path
        
        elif self.agent_name == 'td3_bc':
            # TD3_BCTrainConfig.__post_init__()
            TD3_BCTrainConfig.exp_name = self.exp_name
            name = f"{'td3+bc'}-{self.exp_name}"
            checkpoints_path = os.path.join(weights_folder, name)

            if not os.path.exists(checkpoints_path):
                os.makedirs(checkpoints_path)

            return TD3_BCTrainConfig, checkpoints_path


@dataclass
class IQLTrainConfig:
    device: str = "cuda"
    eval_freq: int = int(100)  # How often (time steps) we evaluate
    n_episodes: int = 1  # How many episodes run during evaluation
    max_timesteps: int = int(1e6)  # Max time steps to run environment
    load_model: str = ""  # Model load file name, "" doesn't load
    buffer_size: int = 2_000_000  # Replay buffer size
    batch_size: int = 2048  # Batch size for all networks
    discount: float = 0.99  # Discount factor
    tau: float = 0.005  # Target network update rate
    beta: float = 3.0  # Inverse temperature. Small beta -> BC, big beta -> maximizing Q
    iql_tau: float = 0.9  # Coefficient for asymmetric loss
    iql_deterministic: bool = False  # Use deterministic actor
    normalize: bool = False  # Normalize states
    normalize_reward: bool = False  # Normalize reward
    vf_lr: float = 3e-4  # V function learning rate
    qf_lr: float = 3e-4  # Critic learning rate
    actor_lr: float = 3e-4  # Actor learning rate
    actor_dropout: Optional[float] = None  # Adroit uses dropout for policy network
    name: str = "IQL"
    n_samples = 500
    elite_percentage = 0.25
    num_subgoals = 10
    num_iters = 10


@dataclass
class TD3_BCTrainConfig:
    # Experiment
    device: str = "cuda"
    eval_freq: int = int(100)  # How often (time steps) we evaluate
    n_episodes: int = 1  # How many episodes run during evaluation
    max_timesteps: int = int(1e6)  # Max time steps to run environment
    load_model: str = ""  # Model load file name, "" doesn't load
    # TD3
    buffer_size: int = 2_000_000  # Replay buffer size
    batch_size: int = 256  # Batch size for all networks
    discount: float = 0.99  # Discount ffor
    expl_noise: float = 0.1  # Std of Gaussian exploration noise
    tau: float = 0.005  # Target network update rate
    policy_noise: float = 0.2  # Noise added to target actor during critic update
    noise_clip: float = 0.5  # Range to clip target actor noise
    policy_freq: int = 2  # Frequency of delayed actor updates
    # TD3 + BC
    alpha: float = 2.5  # Coefficient for Q function in actor loss
    critic_lr : float = 3e-4
    actor_lr : float = 3e-4
    normalize: bool = False  # Normalize states
    normalize_reward: bool = False  # Normalize reward
    name: str = "TD3_BC"
    n_samples = 500
    elite_percentage = 0.25
    num_subgoals = 10
    num_iters = 10