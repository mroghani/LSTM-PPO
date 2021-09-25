from HyperParameter import HyperParameters
from pickle import DICT
import torch
from torch.nn import functional as F
from typing import Dict
import numpy as np

from torch.functional import Tensor
from LoadAndSave import *

from EnvWrappers import MaskVelocityWrapper, PerturbationWrapper
from TrajectoryDataset import TrajectoryDataset
import time
from torch.utils.tensorboard import SummaryWriter

class Trainer:
    def __init__(self,
                 env_name: str,
                 mask_velocity: bool,
                 experiment_name: str,
                 hp: HyperParameters,
                 asynchronous_environment: bool = False,
                 force_cpu_gather: bool = True,
                 checkpoint_frequency: int = 10,
                 workspace_path: str = './workspace') -> None:
        
        self.hp = hp
        self.env_name = env_name
        self.mask_velocity = mask_velocity
        self.obsv_dim, self.action_dim, self.continuous_action_space = get_env_space(env_name)
        self.base_checkpoint_path = f'{workspace_path}/checkpoints/{experiment_name}/'
        self.checkpoint_frequency = checkpoint_frequency
        
        self.train_device = "cuda" if torch.cuda.is_available() else "cpu"
        self.gather_device = "cuda" if torch.cuda.is_available() and not force_cpu_gather else "cpu"
        self.min_reward_values = torch.full([hp.parallel_rollouts], hp.min_reward)
        self.asynchronous_environment = asynchronous_environment
        self.start_or_resume_from_checkpoint()

        self.best_reward = -1e6
        self.fail_to_improve_count = 0

        # Vector environment manages multiple instances of the environment.
        # A key difference between this and the standard gym environment is it automatically resets.
        # Therefore when the done flag is active in the done vector the corresponding state is the first new state.
        self.env = gym.vector.make(self.env_name, self.hp.parallel_rollouts, asynchronous=self.asynchronous_environment)
        if self.mask_velocity:
            self.env = MaskVelocityWrapper(self.env)
        self.env = PerturbationWrapper(self.env, hp.noise)

        self.writer = SummaryWriter(log_dir=f"{workspace_path}/logs/{experiment_name}")
        self.SAVE_METRICS_TENSORBOARD = True

        RANDOM_SEED = 0
        torch.random.manual_seed(RANDOM_SEED)
        np.random.seed(RANDOM_SEED)
        torch.set_num_threads(8)


    
    def start_or_resume_from_checkpoint(self):
        """
        Create actor, critic, actor optimizer and critic optimizer from scratch
        or load from latest checkpoint if it exists. 
        """
        max_checkpoint_iteration = get_last_checkpoint_iteration(self.base_checkpoint_path)
        
        if max_checkpoint_iteration == 0:
            self.actor = Actor(self.obsv_dim,
                        self.action_dim,
                        continuous_action_space=self.continuous_action_space,
                        hp = self.hp)
            self.critic = Critic(self.obsv_dim, self.hp)
            
            self.actor_optimizer = torch.optim.AdamW(self.actor.parameters(), lr=self.hp.actor_learning_rate)
            self.critic_optimizer = torch.optim.AdamW(self.critic.parameters(), lr=self.hp.critic_learning_rate)
         
        # If max checkpoint iteration is greater than zero initialise training with the checkpoint.
        if max_checkpoint_iteration > 0:
            self.actor, self.critic, self.actor_optimizer, self.critic_optimizer, hp, env_name, env_mask_velocity = load_from_checkpoint(self.base_checkpoint_path, max_checkpoint_iteration, 'cpu')
            
            assert env_name == self.env_name, "To resume training environment must match current settings."
            assert env_mask_velocity == self.mask_velocity, "To resume training model architecture must match current settings."
            assert self.hp == hp, "To resume training hyperparameters must match current settings."
            # We have to move manually move optimizer states to TRAIN_DEVICE manually since optimizer doesn't yet have a "to" method.
            for state in self.actor_optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(self.train_device)

            for state in self.critic_optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(self.train_device)
        self.iteration = max_checkpoint_iteration
    
    
    def calc_discounted_return(self, rewards, discount, final_value):
        """
        Calculate discounted returns based on rewards and discount factor.
        """
        seq_len = len(rewards)
        discounted_returns = torch.zeros(seq_len)
        discounted_returns[-1] = rewards[-1] + discount * final_value
        for i in range(seq_len - 2, -1 , -1):
            discounted_returns[i] = rewards[i] + discount * discounted_returns[i + 1]
        return discounted_returns


    def compute_advantages(self, rewards, values, discount, gae_lambda):
        """
        Compute General Advantage.
        """
        deltas = rewards + discount * values[1:] - values[:-1]
        seq_len = len(rewards)
        advs = torch.zeros(seq_len + 1)
        multiplier = discount * gae_lambda
        for i in range(seq_len - 1, -1 , -1):
            advs[i] = advs[i + 1] * multiplier  + deltas[i]
        return advs[:-1]


    def gather_trajectories(self) ->  Dict[str, torch.Tensor]:
        """
        Gather policy trajectories from gym environment.
        """
        
        # Initialise variables.
        obsv = self.env.reset()
        trajectory_data = {"states": [],
                    "actions": [],
                    "action_probabilities": [],
                    "rewards": [],
                    "true_rewards": [],
                    "values": [],
                    "terminals": [],
                    "actor_hidden_states": [],
                    "actor_cell_states": [],
                    "critic_hidden_states": [],
                    "critic_cell_states": []}
        terminal = torch.ones(self.hp.parallel_rollouts) 

        with torch.no_grad():
            # Reset actor and critic state.
            self.actor.get_init_state(self.hp.parallel_rollouts, self.gather_device)
            self.critic.get_init_state(self.hp.parallel_rollouts, self.gather_device)
            # Take 1 additional step in order to collect the state and value for the final state.
            for i in range(self.hp.rollout_steps):
                
                trajectory_data["actor_hidden_states"].append(self.actor.hidden_cell[0].squeeze(0).cpu())
                trajectory_data["actor_cell_states"].append(self.actor.hidden_cell[1].squeeze(0).cpu())
                trajectory_data["critic_hidden_states"].append(self.critic.hidden_cell[0].squeeze(0).cpu())
                trajectory_data["critic_cell_states"].append(self.critic.hidden_cell[1].squeeze(0).cpu())
                
                # Choose next action 
                state = torch.tensor(obsv, dtype=torch.float32)
                trajectory_data["states"].append(state)
                value = self.critic(state.unsqueeze(0).to(self.gather_device), terminal.to(self.gather_device))
                trajectory_data["values"].append( value.squeeze(1).cpu())
                action_dist = self.actor(state.unsqueeze(0).to(self.gather_device), terminal.to(self.gather_device))
                action = action_dist.sample().reshape(self.hp.parallel_rollouts, -1)
                if not self.actor.continuous_action_space:
                    action = action.squeeze(1)
                trajectory_data["actions"].append(action.cpu())
                trajectory_data["action_probabilities"].append(action_dist.log_prob(action).cpu())

                # Step environment 
                action_np = action.cpu().numpy()
                obsv, reward, done, _ = self.env.step(action_np)
                terminal = torch.tensor(done).float()
                transformed_reward = self.hp.scale_reward * torch.max(self.min_reward_values, torch.tensor(reward).float())
                                                                
                trajectory_data["rewards"].append(transformed_reward)
                trajectory_data["true_rewards"].append(torch.tensor(reward).float())
                trajectory_data["terminals"].append(terminal)
        
            # Compute final value to allow for incomplete episodes.
            state = torch.tensor(obsv, dtype=torch.float32)
            value = self.critic(state.unsqueeze(0).to(self.gather_device), terminal.to(self.gather_device))
            # Future value for terminal episodes is 0.
            trajectory_data["values"].append(value.squeeze(1).cpu() * (1 - terminal))

        # Combine step lists into tensors.
        trajectory_tensors = {key: torch.stack(value) for key, value in trajectory_data.items()}
        return trajectory_tensors


    def split_trajectories_episodes(self, trajectory_tensors: Dict[str, torch.Tensor]):
        """
        Split trajectories by episode.
        """

        len_episodes = []
        trajectory_episodes = {key: [] for key in trajectory_tensors.keys()}
        for i in range(self.hp.parallel_rollouts):
            terminals_tmp = trajectory_tensors["terminals"].clone()
            terminals_tmp[0, i] = 1
            terminals_tmp[-1, i] = 1
            split_points = (terminals_tmp[:, i] == 1).nonzero() + 1

            split_lens = split_points[1:] - split_points[:-1]
            split_lens[0] += 1
            
            len_episode = [split_len.item() for split_len in split_lens]
            len_episodes += len_episode
            for key, value in trajectory_tensors.items():
                # Value includes additional step.
                if key == "values":
                    value_split = list(torch.split(value[:, i], len_episode[:-1] + [len_episode[-1] + 1]))
                    # Append extra 0 to values to represent no future reward.
                    for j in range(len(value_split) - 1):
                        value_split[j] = torch.cat((value_split[j], torch.zeros(1)))
                    trajectory_episodes[key] += value_split
                else:
                    trajectory_episodes[key] += torch.split(value[:, i], len_episode)
        return trajectory_episodes, len_episodes


    def pad_and_compute_returns(self, trajectory_episodes, len_episodes):

        """
        Pad the trajectories up to hp.rollout_steps so they can be combined in a
        single tensor.
        Add advantages and discounted_returns to trajectories.
        """

        episode_count = len(len_episodes)
        advantages_episodes, discounted_returns_episodes = [], []
        padded_trajectories = {key: [] for key in trajectory_episodes.keys()}
        padded_trajectories["advantages"] = []
        padded_trajectories["discounted_returns"] = []

        for i in range(episode_count):
            single_padding = torch.zeros(self.hp.rollout_steps - len_episodes[i])
            for key, value in trajectory_episodes.items():
                if value[i].ndim > 1:
                    padding = torch.zeros(self.hp.rollout_steps - len_episodes[i], value[0].shape[1], dtype=value[i].dtype)
                else:
                    padding = torch.zeros(self.hp.rollout_steps - len_episodes[i], dtype=value[i].dtype)
                padded_trajectories[key].append(torch.cat((value[i], padding)))
            padded_trajectories["advantages"].append(torch.cat((self.compute_advantages(rewards=trajectory_episodes["rewards"][i],
                                                            values=trajectory_episodes["values"][i],
                                                            discount=self.hp.discount,
                                                            gae_lambda=self.hp.gae_lambda), single_padding)))
            padded_trajectories["discounted_returns"].append(torch.cat((self.calc_discounted_return(rewards=trajectory_episodes["rewards"][i],
                                                                        discount=self.hp.discount,
                                                                        final_value=trajectory_episodes["values"][i][-1]), single_padding)))
        return_val = {k: torch.stack(v) for k, v in padded_trajectories.items()} 
        return_val["seq_len"] = torch.tensor(len_episodes)
        
        return return_val 


    def train(self):
        
        
        while self.iteration < self.hp.max_iterations:      

            self.actor = self.actor.to(self.gather_device)
            self.critic = self.critic.to(self.gather_device)
            start_gather_time = time.time()

            # Gather trajectories.
            trajectory_tensors = self.gather_trajectories()
            trajectory_episodes, len_episodes = self.split_trajectories_episodes(trajectory_tensors)
            trajectories = self.pad_and_compute_returns(trajectory_episodes, len_episodes)

            # Calculate mean reward.
            complete_episode_count = trajectories["terminals"].sum().item()
            terminal_episodes_rewards = (trajectories["terminals"].sum(axis=1) * trajectories["true_rewards"].sum(axis=1)).sum()
            mean_reward =  terminal_episodes_rewards / complete_episode_count

            # Check stop conditions.
            if mean_reward > self.best_reward:
                self.best_reward = mean_reward
                self.fail_to_improve_count = 0
            else:
                self.fail_to_improve_count += 1
            
            if self.fail_to_improve_count > self.hp.patience:
                print(f"Policy has not yielded higher reward for {self.hp.patience} iterations...  Stopping now.")
                break

            trajectory_dataset = TrajectoryDataset(trajectories, batch_size=self.hp.batch_size,
                                            device=self.train_device, batch_len=self.hp.recurrent_seq_len, rollout_steps=self.hp.rollout_steps)
            end_gather_time = time.time()
            start_train_time = time.time()
            
            self.actor = self.actor.to(self.train_device)
            self.critic = self.critic.to(self.train_device)

            # Train actor and critic. 
            for epoch_idx in range(self.hp.ppo_epochs): 
                for batch in trajectory_dataset:

                    # Get batch 
                    self.actor.hidden_cell = (batch.actor_hidden_states[:1], batch.actor_cell_states[:1])
                    
                    # Update actor
                    self.actor_optimizer.zero_grad()
                    action_dist = self.actor(batch.states)
                    # Action dist runs on cpu as a workaround to CUDA illegal memory access.
                    action_probabilities = action_dist.log_prob(batch.actions[-1, :].to("cpu")).to(self.train_device)
                    # Compute probability ratio from probabilities in logspace.
                    probabilities_ratio = torch.exp(action_probabilities - batch.action_probabilities[-1, :])
                    surrogate_loss_0 = probabilities_ratio * batch.advantages[-1, :]
                    surrogate_loss_1 =  torch.clamp(probabilities_ratio, 1. - self.hp.ppo_clip, 1. + self.hp.ppo_clip) * batch.advantages[-1, :]
                    surrogate_loss_2 = action_dist.entropy().to(self.train_device)
                    actor_loss = -torch.mean(torch.min(surrogate_loss_0, surrogate_loss_1)) - torch.mean(self.hp.entropy_factor * surrogate_loss_2)
                    actor_loss.backward() 
                    torch.nn.utils.clip_grad.clip_grad_norm_(self.actor.parameters(), self.hp.max_grad_norm)
                    self.actor_optimizer.step()

                    # Update critic
                    self.critic_optimizer.zero_grad()
                    self.critic.hidden_cell = (batch.critic_hidden_states[:1], batch.critic_cell_states[:1])
                    values = self.critic(batch.states)
                    critic_loss = F.mse_loss(batch.discounted_returns[-1, :], values.squeeze(1))
                    torch.nn.utils.clip_grad.clip_grad_norm_(self.critic.parameters(), self.hp.max_grad_norm)
                    critic_loss.backward() 
                    self.critic_optimizer.step()
                    
            end_train_time = time.time()
            print(f"Iteration: {self.iteration},  Mean reward: {mean_reward}, Mean Entropy: {torch.mean(surrogate_loss_2)}, " +
                f"complete_episode_count: {complete_episode_count}, Gather time: {end_gather_time - start_gather_time:.2f}s, " +
                f"Train time: {end_train_time - start_train_time:.2f}s")

            if self.SAVE_METRICS_TENSORBOARD:
                self.writer.add_scalar("complete_episode_count", complete_episode_count, self.iteration)
                self.writer.add_scalar("total_reward", mean_reward , self.iteration)
                self.writer.add_scalar("actor_loss", actor_loss, self.iteration)
                self.writer.add_scalar("critic_loss", critic_loss, self.iteration)
                self.writer.add_scalar("policy_entropy", torch.mean(surrogate_loss_2), self.iteration)
            # if SAVE_PARAMETERS_TENSORBOARD:
            #     save_parameters(writer, "actor", actor, iteration)
            #     save_parameters(writer, "value", critic, iteration)
            if self.iteration % self.checkpoint_frequency == 0: 
                save_checkpoint(self.base_checkpoint_path, self.actor, self.critic, self.actor_optimizer, self.critic_optimizer, self.iteration, self.hp, self.env_name, self.mask_velocity)
            self.iteration += 1
            
        return self.best_reward 