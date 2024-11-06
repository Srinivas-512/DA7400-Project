# source: https://github.com/young-geng/CQL/tree/934b0e8354ca431d6c083c4e3a29df88d4b0a24d
# https://arxiv.org/pdf/2006.04779.pdf


import d4rl
import gym
import torch
import os, sys
import numpy as np
from typing import List
from pathlib import Path

parent_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_path)

from Agents import IQL, TD3_BC
from utilities import utils, rbuffer
from CVAE.cvae import *
from plan import *
import argparse
from config import Config

TensorBatch = List[torch.Tensor]

parser = argparse.ArgumentParser()

parser.add_argument("--agent_type", type=str, required=True, help="Low Level Agent Choice (iql or td3_bc)")
parser.add_argument("--exp_name", type=str, required=True, help="Experiment Name")
parser.add_argument("--env", type=str, required=True, help="Environment")
parser.add_argument("--latent_dim", type=int, required=True, help="Latent Dimension of CVAE")
parser.add_argument("--hidden_dim", type=int, required=True, help="Hidden Dimension of CVAE")
parser.add_argument("--subgoal_patience", type=int, required=True, help="Subgoal Patience of CVAE")
parser.add_argument("--eval_method", type=str, required=True, help="Planner Type (higoc / distance_higoc)")
parser.add_argument("--seed", type=int, default=0, required=False, help="Seed")

args = parser.parse_args()

current_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.abspath(os.path.join(current_dir, ".."))
cvae_path = os.path.join(project_dir, "CVAE", "weights", f"{args.env}_{args.latent_dim}_{args.hidden_dim}.pth")

config, checkpoints_path = Config(args.agent_type, args.exp_name).config


def train(config):
    env = gym.make(args.env)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    subgoal_dim = state_dim

    dataset = d4rl.qlearning_dataset(env)

    if config.normalize_reward:
        utils.modify_reward(
            dataset,
            config.env,
            reward_scale=config.reward_scale,
            reward_bias=config.reward_bias,
        )

    if config.normalize:
        state_mean, state_std = utils.compute_mean_std(dataset["observations"], eps=1e-3)
    else:
        state_mean, state_std = 0, 1

    dataset["observations"] = utils.normalize_states(
        dataset["observations"], state_mean, state_std
    )
    dataset["next_observations"] = utils.normalize_states(
        dataset["next_observations"], state_mean, state_std
    )
    env = utils.wrap_env(env, state_mean=state_mean, state_std=state_std)
    replay_buffer = rbuffer.ReplayBuffer(
        state_dim,
        action_dim,
        config.buffer_size,
        config.device,
    )
    replay_buffer.load_d4rl_dataset(dataset)

    max_action = float(env.action_space.high[0])

    # Set seeds
    seed = args.seed
    utils.set_seed(seed, env)

    if args.agent_type == 'iql':

        q_network = IQL.q_funcs.TwinQ(state_dim + subgoal_dim, action_dim).to(config.device)
        v_network = IQL.q_funcs.ValueFunction(state_dim + subgoal_dim).to(config.device)
        actor = (
            IQL.policy.DeterministicPolicy(
                state_dim + subgoal_dim, action_dim, max_action, dropout=config.actor_dropout
            )
            if config.iql_deterministic
            else IQL.policy.GaussianPolicy(
                state_dim + subgoal_dim, action_dim, max_action, dropout=config.actor_dropout
            )
        ).to(config.device)
        v_optimizer = torch.optim.Adam(v_network.parameters(), lr=config.vf_lr)
        q_optimizer = torch.optim.Adam(q_network.parameters(), lr=config.qf_lr)
        actor_optimizer = torch.optim.Adam(actor.parameters(), lr=config.actor_lr)

        kwargs = {
            "max_action": max_action,
            "actor": actor,
            "actor_optimizer": actor_optimizer,
            "q_network": q_network,
            "q_optimizer": q_optimizer,
            "v_network": v_network,
            "v_optimizer": v_optimizer,
            "discount": config.discount,
            "tau": config.tau,
            "device": config.device,
            # IQL
            "beta": config.beta,
            "iql_tau": config.iql_tau,
            "max_steps": config.max_timesteps,
        }

        trainer = IQL.iql.ImplicitQLearning(**kwargs)
    
    if args.agent_type == 'td3_bc':
        actor = TD3_BC.policy.Actor(state_dim+subgoal_dim, action_dim, max_action).to(config.device)
        actor_optimizer = torch.optim.Adam(actor.parameters(), lr=config.actor_lr)

        critic_1 = TD3_BC.q_funcs.Critic(state_dim+subgoal_dim, action_dim).to(config.device)
        critic_1_optimizer = torch.optim.Adam(critic_1.parameters(), lr=config.critic_lr)
        critic_2 = TD3_BC.q_funcs.Critic(state_dim+subgoal_dim, action_dim).to(config.device)
        critic_2_optimizer = torch.optim.Adam(critic_2.parameters(), lr=config.critic_lr)

        kwargs = {
            "max_action": max_action,
            "actor": actor,
            "actor_optimizer": actor_optimizer,
            "critic_1": critic_1,
            "critic_1_optimizer": critic_1_optimizer,
            "critic_2": critic_2,
            "critic_2_optimizer": critic_2_optimizer,
            "discount": config.discount,
            "tau": config.tau,
            "device": config.device,
            # TD3
            "policy_noise": config.policy_noise * max_action,
            "noise_clip": config.noise_clip * max_action,
            "policy_freq": config.policy_freq,
            # TD3 + BC
            "alpha": config.alpha,
        }

        trainer = TD3_BC.td3_bc.TD3_BC(**kwargs)

    goal_planner = CVAE(subgoal_dim, args.latent_dim, args.hidden_dim).to(config.device)
    goal_planner.load_state_dict(torch.load(cvae_path))
    goal_planner.eval()
    N = args.subgoal_patience

    
    print("---------------------------------------")
    print(f"Training HiGOC (IQL Based), Env: {args.env}, Seed: {seed}")
    print("---------------------------------------")


    if config.load_model != "":
        policy_file = Path(config.load_model)
        trainer.load_state_dict(torch.load(policy_file))
        actor = trainer.actor

    evaluations = []
    for t in range(int(config.max_timesteps)):
        batch = replay_buffer.sample(config.batch_size)

        states, actions, rewards, next_states, dones, indices, start_indices, end_indices = batch

        try:
            proxy_indices = np.random.randint(np.maximum(indices, start_indices.squeeze(1).cpu().numpy() + N), end_indices.squeeze(1).cpu().numpy()+1, size=len(indices))
            prev_state_proxies = replay_buffer._states[proxy_indices - N]
        except:
            proxy_indices = np.random.randint(start_indices.squeeze(1).cpu().numpy() + 1, end_indices.squeeze(1).cpu().numpy()+1, size=len(indices))
            prev_state_proxies = replay_buffer._states[proxy_indices - 1]

        future_state_proxies = replay_buffer._states[proxy_indices]

        future_state_proxies[:, -(state_dim-2):] = 0.0
        prev_state_proxies[:, -(state_dim-2):] = 0.0

        noisy_subgoals = sample_noisy_subgoal(goal_planner, future_state_proxies, prev_state_proxies, device = config.device)

        distances = torch.norm(noisy_subgoals[:, :2]-states[:, :2], dim=1)
        # distances = torch.norm((noisy_subgoals[:, :2] - states[:, :2]).clone(), dim=1)

        tdm_subgoals = torch.tensor((proxy_indices == indices)).to(config.device)

        distances = torch.where(tdm_subgoals, torch.zeros_like(distances), distances)

        beta = 0.01

        rewards -= beta*distances.unsqueeze(1)
        # rewards -= 1

        observations = torch.cat((states, noisy_subgoals), dim=1)
        next_observations = torch.cat((next_states, noisy_subgoals), dim=1)

        batch = [observations, actions, rewards, next_observations, dones]
        batch = [b.to(config.device) for b in batch]
        
        log_dict = trainer.train(batch)

        # Evaluate episode

        if args.agent_type == 'iql':
            if (t%100==0):
                print(f"Actor loss = {log_dict['actor_loss']} |  Q_loss = {log_dict['q_loss']} |  Value_loss = {log_dict['value_loss']}")
            if (t + 1) % config.eval_freq == 0:
                print(f"Time steps: {t + 1}")
                eval(config, args, goal_planner, v_network, actor, env)
                torch.save(
                    trainer.state_dict(),
                    os.path.join(checkpoints_path, f"checkpoint_{t}.pt"),
                )
        
        if args.agent_type == 'td3_bc':
            if (t%100==0):
                try:
                    print(f"Actor loss = {log_dict['actor_loss']} |  Q_loss = {log_dict['critic_loss']}")
                except:
                    print(f"Q_loss = {log_dict['critic_loss']}")
            if (t + 1) % config.eval_freq == 0:
                print(f"Time steps: {t + 1}")
                eval(config, args, goal_planner, critic_1, actor, env)
                torch.save(
                    trainer.state_dict(),
                    os.path.join(checkpoints_path, f"checkpoint_{t}.pt"),
                )


if __name__ == "__main__":
    train(config)