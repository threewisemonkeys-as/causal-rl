"""
Causal PPO
Atharv Sonwane <atharvs.twm@gmail.com>
Rishab Patra
"""

import torch
from torch.nn import functional as F

from .ppo import PPO

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.double

# Hyperparameters
EPOCHS = 20
EPISODES_PER_EPOCH = 8
POLICY_HIDDEN_LAYERS = 64
VALUE_HIDDEN_LAYERS = 64
N_POLICY_UPDATES = 16
N_VALUE_UPDATES = 16
GAMMA = 0.99
EPSILON = 0.1
VALUE_FN_LEARNING_RATE = 1e-3
POLICY_LEARNING_RATE = 3e-4
MAX_TRAJ_LENGTH = 1000


# Model
class CausalPPO(PPO):
    def __init__(self, compute_causal_factor_fn, *args, **kwargs):
        super(CausalPPO, self).__init__(*args, **kwargs)
        self._compute_causal_factor = compute_causal_factor_fn

    def _update(self, batch, hp, policy_optim, value_optim, epoch, writer):
        # process batch
        obs = [torch.stack(traj.obs)[:-1] for traj in batch]
        disc_r = [traj.disc_r(hp["gamma"], normalize=True) for traj in batch]
        a = [torch.stack(traj.a) for traj in batch]
        with torch.no_grad():
            v = [self.value(o).view(-1) for o in obs]
            adv = [disc_r[i] - v[i] for i in range(len(batch))]
            old_logits = [torch.stack(traj.logits) for traj in batch]
            old_logprobs = [
                -F.cross_entropy(old_logits[i], a[i]) for i in range(len(batch))
            ]
            c = []
            c_info = []
            for i, traj in enumerate(batch):
                causal_factor, causal_info = self._compute_causal_factor(traj)
                c.append(causal_factor)
                c_info.append(causal_info)
                adv[i] *= causal_factor
                for j, cf in enumerate(causal_factor):
                    writer.add_scalar(f"causal_factor/E{epoch+1}T{i+1}", cf, j)
            for j, cf in enumerate(torch.stack(c, dim=0).mean(dim=0)):
                writer.add_scalar(f"causality/E{epoch+1}", cf, j)

        # update policy
        for j in range(hp["n_policy_updates"]):
            policy_loss = torch.zeros(1, device=device, dtype=dtype, requires_grad=True)
            for i, traj in enumerate(batch):
                curr_logits = self.policy(obs[i])
                curr_logprobs = -F.cross_entropy(curr_logits, a[i])
                ratio = torch.exp(curr_logprobs - old_logprobs[i])
                clipped_ratio = torch.clamp(ratio, 1 - hp["epsilon"], 1 + hp["epsilon"])
                policy_loss = (
                    policy_loss
                    + torch.min(ratio * adv[i], clipped_ratio * adv[i]).mean()
                )

            policy_loss = policy_loss / len(batch)
            policy_optim.zero_grad()
            policy_loss.backward()
            policy_optim.step()

        # update value function
        for j in range(hp["n_value_updates"]):
            value_loss = torch.zeros(1, device=device, dtype=dtype, requires_grad=True)
            for i in range(len(batch)):
                v = self.value(obs[i]).view(-1)
                value_loss = value_loss + F.mse_loss(v, disc_r[i])
            value_loss = value_loss / len(batch)
            value_optim.zero_grad()
            value_loss.backward()
            value_optim.step()

        metrics = {
            "loss/policy_loss": policy_loss.item(),
            "loss/value_loss": value_loss.item(),
            "causality/mean_causal_factor": torch.mean(causal_factor),
        }

        return metrics
