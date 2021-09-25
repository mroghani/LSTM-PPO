import os
import pickle
import torch
from typing import Optional
import gym
from Actor import Actor
from Critic import Critic
from dotmap import DotMap
import pathlib


def get_last_checkpoint_iteration(base_checkpoint_path: str) -> int:
    """Determine latest checkpoint iteration.

    Parameters
    ----------
    base_checkpoint_path : str
    """
    if os.path.isdir(base_checkpoint_path):
        max_checkpoint_iteration = max([int(dirname) for dirname in os.listdir(base_checkpoint_path)])
    else:
        max_checkpoint_iteration = 0
    return max_checkpoint_iteration


def load_checkpoint(base_checkpoint_path: str, iteration: int, map_loacation: Optional[str] = None):
    """
    Load from training checkpoint.

    Parameters
    ----------
    base_checkpoint_path : str
    
    itration: int
    
    map_location: str, optional
    """
    base_checkpoint = base_checkpoint_path + f"{iteration}/"
    with open(base_checkpoint + "parameters.pt", "rb") as f:
        checkpoint = pickle.load(f)
        

    actor_state_dict = torch.load(base_checkpoint + "actor.pt", map_location=torch.device(map_loacation))
    critic_state_dict = torch.load(base_checkpoint + "critic.pt", map_location=torch.device(map_loacation))
    actor_optimizer_state_dict = torch.load(base_checkpoint + "actor_optimizer.pt", map_location=torch.device(map_loacation))
    critic_optimizer_state_dict = torch.load(base_checkpoint + "critic_optimizer.pt", map_location=torch.device(map_loacation))
    
    return (actor_state_dict, critic_state_dict,
           actor_optimizer_state_dict, critic_optimizer_state_dict,
           checkpoint.hp, checkpoint.env, checkpoint.env_mask_velocity)


def get_env_space(env_name: str):
    """
    Return obsvervation dimensions, action dimensions and whether or not action space is continuous.
    """
    env = gym.make(env_name)
    continuous_action_space = type(env.action_space) is gym.spaces.box.Box
    if continuous_action_space:
        action_dim =  env.action_space.shape[0]
    else:
        action_dim = env.action_space.n 
    obsv_dim= env.observation_space.shape[0] 
    return obsv_dim, action_dim, continuous_action_space


def load_from_checkpoint(base_checkpoint_path: str, iteration: int, map_loacation: Optional[str] = None):
    
    actor_state_dict, critic_state_dict, actor_optimizer_state_dict, critic_optimizer_state_dict, hp, env_name, env_mask_velocity = load_checkpoint(base_checkpoint_path, iteration, map_loacation)
    
    obsv_dim, action_dim, continuous_action_space = get_env_space(env_name)
    actor = Actor(obsv_dim,
                  action_dim,
                  continuous_action_space=continuous_action_space,
                  hp = hp)
    critic = Critic(obsv_dim, hp)
    
    actor_optimizer = torch.optim.AdamW(actor.parameters(), lr=hp.actor_learning_rate)
    critic_optimizer = torch.optim.AdamW(critic.parameters(), lr=hp.critic_learning_rate)

    actor.load_state_dict(actor_state_dict, strict=True) 
    critic.load_state_dict(critic_state_dict, strict=True)
    actor_optimizer.load_state_dict(actor_optimizer_state_dict)
    critic_optimizer.load_state_dict(critic_optimizer_state_dict)

    # We have to move manually move optimizer states to TRAIN_DEVICE manually since optimizer doesn't yet have a "to" method.
    for state in actor_optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to("cpu")

    for state in critic_optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to("cpu")

    return actor, critic, actor_optimizer, critic_optimizer, hp, env_name, env_mask_velocity



def save_checkpoint(base_checkpoint_path: str, actor, critic, actor_optimizer, critic_optimizer, iteration, hp, env_name, env_mask_velocity):
    """
    Save training checkpoint.
    """
    checkpoint = DotMap()
    checkpoint.env = env_name
    checkpoint.env_mask_velocity = env_mask_velocity 
    checkpoint.iteration = iteration
    checkpoint.hp = hp
    base_checkpoint = base_checkpoint_path + f"{iteration}/"
    pathlib.Path(base_checkpoint).mkdir(parents=True, exist_ok=True) 
    with open(base_checkpoint + "parameters.pt", "wb") as f:
        pickle.dump(checkpoint, f)
    
    # TODO: Remove this
    with open(base_checkpoint + "actor_class.pt", "wb") as f:
        pickle.dump(Actor, f)
    with open(base_checkpoint + "critic_class.pt", "wb") as f:
        pickle.dump(Critic, f)
    # END_TODO
    
    torch.save(actor.state_dict(), base_checkpoint + "actor.pt")
    torch.save(critic.state_dict(), base_checkpoint + "critic.pt")
    torch.save(actor_optimizer.state_dict(), base_checkpoint + "actor_optimizer.pt")
    torch.save(critic_optimizer.state_dict(), base_checkpoint + "critic_optimizer.pt")