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
