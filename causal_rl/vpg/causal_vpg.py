"""
Causal Policy Gradient in pytorch
Atharv Sonwane

References:
https://spinningup.openai.com/en/latest/algorithms/vpg.html
"""

import torch
from torch.nn import functional as F

from .vpg import VPG

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
class CausalPG(VPG):
    def __init__(self, compute_causal_factor_fn, *args, **kwargs):
        super(CausalPG, self).__init__(*args, **kwargs)
        self._compute_causal_factor = compute_causal_factor_fn

    def _update(self, batch, hp, policy_optim, value_optim):
        # process batch
        obs = [torch.stack(traj.obs)[:-1] for traj in batch]
        disc_r = [traj.disc_r(hp["gamma"], normalize=True) for traj in batch]
        a = [torch.stack(traj.a) for traj in batch]
        with torch.no_grad():
            v = [self.value(o).view(-1) for o in obs]
            adv = [disc_r[i] - v[i] for i in range(len(batch))]
            c = []
            c_info = []
            for i, traj in enumerate(batch):
                causal_factor, causal_info = self._compute_causal_factor(traj)
                c.append(causal_factor)
                c_info.append(causal_info)
                adv[i] *= causal_factor

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
            "causality/mean_causal_factor": torch.mean(causal_factor),
        }
