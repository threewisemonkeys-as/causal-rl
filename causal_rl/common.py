import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.double


class Trajectory:
    def __init__(
        self, observations=[], actions=[], rewards=[], dones=[], logits=[],
    ):
        self.obs = observations
        self.a = actions
        self.r = rewards
        self.d = dones
        self.logits = logits
        self.len = 0

    def add(
        self,
        obs: torch.Tensor,
        a: torch.Tensor,
        r: torch.Tensor,
        d: torch.Tensor,
        logits: torch.Tensor,
    ):
        self.obs.append(obs)
        self.a.append(a)
        self.r.append(r)
        self.d.append(d)
        self.logits.append(logits)
        self.len += 1

    def disc_r(self, gamma, normalize=False):
        disc_rewards = []
        r = 0.0
        for reward in self.r[::-1]:
            r = reward + gamma * r
            disc_rewards.insert(0, r)
        disc_rewards = torch.tensor(disc_rewards, device=device, dtype=dtype)
        if normalize:
            disc_rewards = (disc_rewards - disc_rewards.mean()) / (
                disc_rewards.std() + np.finfo(np.float32).eps
            )
        return disc_rewards

    def __len__(self):
        return self.len


def compute_causal_factor1(traj):
    no_drug = 0
    pi_only = 1
    rti_only = 2
    pi_rti = 3
    free_virus = 4

    obs = torch.stack(traj.obs)
    actions = torch.stack(traj.a)
    counts = torch.cumsum(torch.ones(len(traj), dtype=dtype, device=device), dim=0)
    action_tracker = torch.cumsum(F.one_hot(actions), dim=0)
    p_a = (
        action_tracker[:, pi_only]
        + action_tracker[:, rti_only]
        + action_tracker[:, pi_rti]
    ) / counts
    p_not_a_and_b = (
        torch.cumsum((obs[:-1, free_virus] > 415) * (actions == no_drug), dim=0)
        / counts
    )
    p_a_and_b = torch.cumsum((obs[:-1, 4] > 415) * (actions != no_drug), dim=0) / counts
    c = (p_a_and_b / (p_a + 1e-5)) - (p_not_a_and_b / (1 - p_a + 1e-5))
    return c, {"p_a": p_a, "p_not_a_and_b": p_not_a_and_b, "p_a_and_b": p_a_and_b}


def sample_traj(agent, env, max_timesteps):
    # initialise tracking variables
    obs = env.reset()
    obs = torch.tensor(obs, device=device, dtype=dtype)
    traj = Trajectory([obs], [], [], [], [])
    d = False

    # run for single trajectory
    for _ in range(max_timesteps):
        a = agent.act(obs)
        obs, r, d, _ = env.step(a)
        a = torch.tensor(a)
        obs = torch.tensor(obs)
        r = torch.tensor(r)
        traj.add(obs, a, r, d, None)

        if d:
            break

    return traj


def plot_agent_behaviors(agents, env, state_names, max_timesteps, save_path=None):
    fig, axes = plt.subplots(5, 2, sharex=True, figsize=[7, 8])
    axes = axes.flatten()

    for name, agent in agents.items():
        traj = sample_traj(agent, env, max_timesteps)
        obs = torch.stack(traj.obs, dim=0)
        for i, sname in enumerate(state_names):
            y = torch.log(obs[:, i])
            axes[i].plot(y, label=name)
            axes[i].set_title(f"log_{sname}")

        action = torch.stack(traj.a, dim=0)
        epsilon_1 = ((action == 2) | (action == 3)).to(float) * 0.7
        epsilon_2 = ((action == 1) | (action == 3)).to(float) * 0.3
        axes[-3].plot(epsilon_1, label=name)
        axes[-3].set_title("Treatment epsilon_1")
        axes[-2].plot(epsilon_2, label=name)
        axes[-2].set_title("Treatment epsilon_2")

        reward = torch.stack(traj.r, dim=0)
        axes[-1].plot(reward, label=name)
        axes[-1].set_title("reward")

    for ax in axes:
        ax.legend()
    fig.tight_layout(pad=0.3)
    if save_path is not None:
        plt.savefig(save_path)
    plt.show()
