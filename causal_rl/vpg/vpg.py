"""
Policy Gradient in pytorch
Atharv Sonwane

References:
https://spinningup.openai.com/en/latest/algorithms/vpg.html
"""

import datetime
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from causal_rl.common import Trajectory

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.double

# Hyperparameters
EPOCHS = 100
EPISODES_PER_EPOCH = 16
POLICY_HIDDEN_LAYERS = 64
VALUE_HIDDEN_LAYERS = 64
GAMMA = 0.99
VALUE_FN_LEARNING_RATE = 1e-2
POLICY_LEARNING_RATE = 1e-3
MAX_TRAJ_LENGTH = 1000


# Model
class VPG:
    def __init__(
        self,
        env,
        policy_hidden_layers=POLICY_HIDDEN_LAYERS,
        value_hidden_layers=VALUE_HIDDEN_LAYERS,
    ):
        self.env = env
        if self.env.unwrapped.spec is not None:
            self.env_name = self.env.unwrapped.spec.id
        else:
            self.env_name = self.env.unwrapped.__class__.__name__
        self.policy = (
            nn.Sequential(
                nn.Linear(env.observation_space.shape[0], policy_hidden_layers),
                nn.Dropout(p=0.6),
                nn.ReLU(),
                nn.Linear(policy_hidden_layers, env.action_space.n),
            )
            .to(device)
            .to(dtype)
        )
        self.value = (
            nn.Sequential(
                nn.Linear(env.observation_space.shape[0], value_hidden_layers),
                nn.Dropout(p=0.6),
                nn.ReLU(),
                nn.Linear(value_hidden_layers, 1),
            )
            .to(device)
            .to(dtype)
        )

    def _update(self, batch, hp, policy_optim, value_optim):
        # process batch
        obs = [torch.stack(traj.obs)[:-1] for traj in batch]
        disc_r = [traj.disc_r(hp["gamma"], normalize=True) for traj in batch]
        a = [torch.stack(traj.a) for traj in batch]
        with torch.no_grad():
            v = [self.value(o) for o in obs]
            adv = [disc_r[i] - v[i] for i in range(len(batch))]

        # update policy
        policy_loss = torch.zeros(1, device=device, dtype=dtype, requires_grad=True)
        for i, traj in enumerate(batch):
            curr_logits = self.policy(obs[i])
            curr_logprobs = -F.cross_entropy(curr_logits, a[i])
            policy_loss = policy_loss + -1 * torch.sum(curr_logprobs * adv[i])

        policy_loss = policy_loss / len(batch)
        policy_optim.zero_grad()
        policy_loss.backward()
        policy_optim.step()

        # update value function
        value_loss = torch.zeros(1, device=device, dtype=dtype, requires_grad=True)
        for i in range(len(batch)):
            v = self.value(obs[i]).view(-1)
            value_loss = value_loss + F.mse_loss(v, disc_r[i])
        value_loss = value_loss / len(batch)
        value_optim.zero_grad()
        value_loss.backward()
        value_optim.step()

        return {
            "loss/policy_loss": policy_loss.item(),
            "loss/value_loss": value_loss.item(),
        }

    def train(
        self,
        epochs=EPOCHS,
        episodes_per_epoch=EPISODES_PER_EPOCH,
        value_lr=VALUE_FN_LEARNING_RATE,
        policy_lr=POLICY_LEARNING_RATE,
        gamma=GAMMA,
        max_traj_length=MAX_TRAJ_LENGTH,
        log_dir="./logs/",
        RENDER=False,
        PLOT_REWARDS=True,
        VERBOSE=False,
        TENSORBOARD_LOG=True,
    ):
        """ Trains both policy and value networks """
        hp = locals()
        start_time = datetime.datetime.now()
        print(
            f"Start time: {start_time:%d-%m-%Y %H:%M:%S}"
            f"\nTraining model on {self.env_name} | "
            f"Observation Space: {self.env.observation_space} | "
            f"Action Space: {self.env.action_space}\n"
            f"Hyperparameters: \n{hp}\n"
        )
        log_path = Path(log_dir).joinpath(f"{start_time:%d%m%Y%H%M%S}")
        log_path.mkdir(parents=True, exist_ok=False)
        if TENSORBOARD_LOG:
            writer = SummaryWriter(log_path)
            writer.add_text("hyperparameters", f"{hp}")
        else:
            writer = None

        self.policy.train()
        self.value.train()
        value_optim = torch.optim.Adam(self.value.parameters(), lr=value_lr)
        policy_optim = torch.optim.Adam(self.policy.parameters(), lr=policy_lr)
        rewards = []
        e = 0

        try:
            for epoch in range(epochs):

                epoch_rewards = []
                batch = []

                # Sample trajectories
                for _ in range(episodes_per_epoch):
                    # initialise tracking variables
                    obs = self.env.reset()
                    obs = torch.tensor(obs, device=device, dtype=dtype)
                    traj = Trajectory([obs], [], [], [], [])
                    d = False
                    e += 1

                    # run for single trajectory
                    for i in range(max_traj_length):
                        if RENDER and (
                            e == 0 or (e % ((epochs * episodes_per_epoch) / 10)) == 0
                        ):
                            self.env.render()

                        a_logits = self.policy(obs)
                        a = torch.distributions.Categorical(logits=a_logits).sample()

                        obs, r, d, _ = self.env.step(a.item())

                        obs = torch.tensor(obs, device=device, dtype=dtype)
                        r = torch.tensor(r, device=device, dtype=dtype)
                        traj.add(obs, a, r, d, a_logits)

                        if d:
                            break

                    epoch_rewards.append(sum(traj.r).to("cpu").numpy())
                    batch.append(traj)

                # Update value and policy
                metrics = self._update(batch, hp, policy_optim, value_optim)

                # Log rewards and losses
                metrics["avg_episode_reward"] = np.mean(
                    epoch_rewards[-episodes_per_epoch:]
                )
                rewards.append(metrics["avg_episode_reward"])
                if writer is not None:
                    for key, val in metrics.items():
                        writer.add_scalar(key, val, epoch)

                if VERBOSE and (epoch == 0 or ((epoch + 1) % (epochs / 10)) == 0):
                    r = metrics["avg_episode_reward"]
                    p_loss = metrics["loss/policy_loss"]
                    v_loss = metrics["loss/value_loss"]
                    print(
                        f"Epoch {epoch + 1}: Average Episodic Reward = {r:.2f} |"
                        f" Value Loss = {p_loss:.2f} |"
                        f" Policy Loss = {v_loss:.2f}"
                    )

        except KeyboardInterrupt:
            print("\nTraining Interrupted!\n")

        finally:
            self.env.close()
            print(
                f"\nTraining Completed in {(datetime.datetime.now() - start_time).seconds} seconds"
            )
            self.save(
                log_path.joinpath(f"{self.__class__.__name__}_{self.env_name}.pt")
            )
            if PLOT_REWARDS:
                plt.plot(rewards)
                plt.savefig(
                    log_path.joinpath(
                        f"{self.__class__.__name__}_{self.env_name}_reward_plot.png"
                    )
                )

    def act(self, obs):
        self.policy.eval()
        with torch.no_grad():
            obs = torch.tensor(obs).to(device).to(dtype)
            logits = self.policy(obs)
        return torch.distributions.Categorical(logits=logits).sample().item()

    def save(self, path=None):
        """ Save model parameters """
        if path is None:
            path = f"{self.__class__.__name__}_{self.env_name}.pt"
        torch.save(
            {
                "policy_state_dict": self.policy.state_dict(),
                "value_state_dict": self.value.state_dict(),
            },
            path,
        )
        print(f"\nSaved model parameters to {path}")

    def load(self, path=None):
        """ Load model parameters """
        if path is None:
            path = f"{self.__class__.__name__}_{self.env_name}.pt"
        checkpoint = torch.load(path)
        self.policy.load_state_dict(checkpoint["policy_state_dict"])
        self.value.load_state_dict(checkpoint["value_state_dict"])
        print(f"\nLoaded model parameters from {path}")

    def eval(self, episodes, render=False):
        """ Evaluates model performance """

        print(f"\nEvaluating model for {episodes} episodes ...\n")
        start_time = time.time()
        self.policy.eval()
        rewards = []

        for episode in range(episodes):

            observation = self.env.reset()
            observation = torch.tensor(observation, device=device, dtype=dtype)
            done = False
            episode_rewards = []

            while not done:
                if render:
                    self.env.render()

                action_probs = self.policy(observation)
                action_distribution = torch.distributions.Categorical(action_probs)
                action = action_distribution.sample()

                next_observation, reward, done, _ = self.env.step(action.item())
                episode_rewards.append(float(reward))
                next_observation = torch.tensor(
                    next_observation, device=device, dtype=dtype
                )
                observation = next_observation

            total_episode_reward = sum(episode_rewards)
            rewards.append(total_episode_reward)
            print(
                f"Episode {episode+1}: Total Episode Reward = {total_episode_reward:.2f}"
            )
            rewards.append(total_episode_reward)

        self.env.close()
        print(f"\nAverage Reward for an episode = {np.mean(rewards):.2f}")
        print(f"Evaluation Completed in {(time.time() - start_time):.2f} seconds")
