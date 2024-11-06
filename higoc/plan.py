import gym
import d4rl
import os, sys
import numpy as np
from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F

agent_modules_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Agents'))
sys.path.append(agent_modules_path)

TensorBatch = List[torch.Tensor]

min_scale_value = 1e-6


def sample_latent(
        n_samples:int, mu:torch.tensor, std:torch.tensor) -> torch.tensor:
    std = torch.clamp(std, min=min_scale_value)
    distribution = torch.distributions.Normal(mu, std)
    samples = distribution.sample([n_samples]) 
    log_probs = distribution.log_prob(samples)
    return samples, log_probs


@torch.no_grad()
def plan_objective_higoc(
        config, args, latents:torch.tensor, goal_planner:nn.Module, actor:nn.Module, value_function:nn.Module, start_state:torch.tensor, true_goal:torch.tensor, num_subgoals:int, log_probs:torch.Tensor, mu, std
) -> float:
    device = config.device
    start_state = torch.tensor(start_state, dtype=torch.float32).to(device)
    g = torch.zeros(config.n_samples, num_subgoals, start_state.shape[0])
    
    g[:,0] = goal_planner.decoder(start_state.unsqueeze(0).repeat(config.n_samples, 1),latents)
    states = torch.zeros(config.n_samples, num_subgoals, start_state.shape[0])
    states[:, 0] = start_state
    
    for i in range(1, num_subgoals-1):
        new_latents, log_probs_current = sample_latent(len(log_probs), mu, std)
        log_probs += torch.sum(log_probs_current, dim=1).to(device)
        goal = goal_planner.decoder(g[:, i-1].to(device), new_latents.to(device))
        g[:, i] = goal
        states[:, i] = g[:,  i-1]
    
    g[:,-1] = F.pad(true_goal, (0, start_state.shape[0]-len(true_goal)), "constant", 0).unsqueeze(0).repeat(config.n_samples, 1)
    states[:, -1] = g[:, -2]
    
    observations = torch.cat((states, g), dim = -1).to(device)

    if args.agent_type == 'iql':
        values = value_function(observations)
        values = torch.sum(values, dim = 1)
    
    if args.agent_type == 'td3_bc':
        actions = actor(observations)
        values = value_function(observations, actions)
        values = torch.sum(values.squeeze(2), dim = 1)

    return values - 0.1*log_probs

@torch.no_grad()
def plan_objective_distance_higoc(
        config, args, latents:torch.tensor, goal_planner:nn.Module, actor:nn.Module, value_function:nn.Module, start_state:torch.tensor, true_goal:torch.tensor, num_subgoals:int, log_probs:torch.Tensor, mu, std
) -> float:
    device = config.device
    start_state = torch.tensor(start_state, dtype=torch.float32).to(device)
    g = torch.zeros(config.n_samples, num_subgoals, start_state.shape[0])
    
    g[:,0] = goal_planner.decoder(start_state.unsqueeze(0).repeat(config.n_samples, 1),latents)
    goal_tensor = true_goal.unsqueeze(0).repeat(config.n_samples, 1)
    states = torch.zeros(config.n_samples, num_subgoals, start_state.shape[0])
    states[:, 0] = start_state
    total_distance = torch.zeros(config.n_samples,num_subgoals, device=device)

    for i in range(1, num_subgoals-1):
        new_latents, log_probs_current = sample_latent(len(log_probs), mu, std)
        log_probs += torch.sum(log_probs_current, dim=1).to(device)
        goal = goal_planner.decoder(g[:, i-1].to(device), new_latents.to(device))
        g[:, i] = goal
        states[:, i] = g[:, i-1]
        total_distance[:,i]=torch.norm(goal[:, :2]-goal_tensor.to(device), dim=1)

    g[:,-1] = F.pad(true_goal, (0, start_state.shape[0]-len(true_goal)), "constant", 0).unsqueeze(0).repeat(config.n_samples, 1)
    states[:, -1] = g[:, -2]
    
    observations = torch.cat((states, g), dim = -1).to(device)

    if args.agent_type == 'iql':
        values = value_function(observations)
    
    if args.agent_type == 'td3_bc':
        actions = actor(observations)
        values = value_function(observations, actions)
        values = values.squeeze(2)

    values_new = values+0.1*total_distance

    return torch.sum(values_new, dim=1) - 0.1*log_probs 


def CEM(
    args, config, K:int, start_state:torch.tensor, true_goal:torch.tensor, num_subgoals:int, actor:nn.Module, goal_planner:nn.Module, value_function:nn.Module
) -> np.ndarray:
    device = config.device
    plan_method = args.eval_method
    mu = torch.zeros(args.latent_dim)
    std = torch.ones(args.latent_dim)
    start_state = torch.tensor(start_state, dtype=torch.float32).to(device)

    for iter in range(config.num_iters):
        latents, log_probs = sample_latent(config.n_samples, mu, std)
        latents = latents.to(device)
        log_probs = torch.sum(log_probs, dim=1).to(device)

        if plan_method == 'higoc':
            objectives = plan_objective_higoc(config, args, latents, goal_planner, actor, value_function, start_state, true_goal, num_subgoals, log_probs, mu, std)
        
        if plan_method == 'distance_higoc':
            objectives = plan_objective_distance_higoc(config, args, latents, goal_planner, actor, value_function, start_state, true_goal, num_subgoals, log_probs, mu, std)

        _, elite_indices = torch.topk(objectives, int(K))
        mu = torch.mean(latents[elite_indices], dim=0)
        std = torch.std(latents[elite_indices], dim=0)

    
    latents, _ = sample_latent(1, mu, std)

    goal = goal_planner.decoder(start_state,latents.squeeze(0)).squeeze(0)

    return goal


@torch.no_grad()
def eval_actor_higoc(
    config, args, env: gym.Env, actor: nn.Module, goal_planner: nn.Module, value_function: nn.Module
) -> np.ndarray:
    env.seed(args.seed)
    actor.eval()
    device = config.device
    episode_rewards = []
    for _ in range(config.n_episodes):
        state, done = env.reset(), False
        true_goal = torch.tensor(env.target_goal, dtype=torch.float32).to(device)
        episode_reward = 0.0
        tij = 0
        subgoal = CEM(args, config, config.elite_percentage*config.n_samples, state, true_goal, config.num_subgoals, actor, goal_planner, value_function)
        while not done:
            state = torch.tensor(state, dtype=torch.float32).to(device)
            if tij>args.subgoal_patience or torch.norm(subgoal[:2]-state[:2]).item() <= 1.5:
                subgoal = CEM(args, config, config.elite_percentage*config.n_samples, state, true_goal, config.num_subgoals, actor, goal_planner, value_function)
                tij = 0
            
            action = actor.act(torch.cat((state, subgoal), dim = 0), device)
            state, reward, done, _ = env.step(action)
            episode_reward += reward
            tij += 1
        episode_rewards.append(episode_reward)
    actor.train()
    return np.asarray(episode_rewards)

@torch.no_grad()
def eval_actor_distance_higoc(
    config, args, env: gym.Env, actor: nn.Module, goal_planner: nn.Module, value_function: nn.Module
) -> np.ndarray:
    env.seed(args.seed)
    device = config.device
    actor.eval()
    episode_rewards = []
    for _ in range(config.n_episodes):
        num_subgoals = config.num_subgoals
        state, done = env.reset(), False
        true_goal = torch.tensor(env.target_goal, dtype=torch.float32).to(device)
        true_goal_expanded = torch.cat((true_goal, torch.zeros(env.observation_space.shape[0]-2).to(device)))
        episode_reward = 0.0
        tij = 0
        subgoal = CEM(args, config, config.elite_percentage*config.n_samples, state, true_goal, num_subgoals, actor, goal_planner, value_function)
        num_subgoals -=1
        while not done:
            state = torch.tensor(state, dtype=torch.float32).to(device)
            if num_subgoals<=2:
                subgoal = true_goal_expanded

            else:
                if tij>args.subgoal_patience or torch.norm(subgoal[:2]-state[:2]).item() <= 1.5:
                    subgoal = CEM(args, config, config.elite_percentage*config.n_samples, state, true_goal, num_subgoals, actor, goal_planner, value_function)
                    tij = 0
                    num_subgoals -=1
            
            action = actor.act(torch.cat((state, subgoal), dim = 0), device)
            state, reward, done, _ = env.step(action)
            episode_reward += reward
            tij += 1
        episode_rewards.append(episode_reward)
    actor.train()
    return np.asarray(episode_rewards)


def eval(config, args, goal_planner, value_function, actor, env):
    
    evaluations = []

    if args.eval_method == 'higoc':

        eval_scores = eval_actor_higoc(
                    config, 
                    args,
                    env,
                    actor,
                    goal_planner, 
                    value_function
                )
        eval_score = eval_scores.mean()
        normalized_eval_score = env.get_normalized_score(eval_score) * 100.0
        evaluations.append(normalized_eval_score)
        print("---------------------------------------")
        print(
            f"Evaluation over {config.n_episodes} episodes: "
            f"{eval_score:.5f} , D4RL score: {normalized_eval_score:.5f}"
        )
        print("---------------------------------------")
    
    if args.eval_method == 'distance_higoc':

        eval_scores = eval_actor_distance_higoc(
                    config,
                    args,
                    env,
                    actor,
                    goal_planner, 
                    value_function
                )
        eval_score = eval_scores.mean()
        normalized_eval_score = env.get_normalized_score(eval_score) * 100.0
        evaluations.append(normalized_eval_score)
        print("---------------------------------------")
        print(
            f"Evaluation over {config.n_episodes} episodes: "
            f"{eval_score:.5f} , D4RL score: {normalized_eval_score:.5f}"
        )
        print("---------------------------------------")



if __name__ == "__main__":
    eval()