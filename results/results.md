# Results

## Experiment

Policies were trained on HIV simulator through standard RL setup. Both regular viersions as well as modified *causal* versions were tried. The Causal versions of each algorithm involve calculating a causal factor which is used to scale the advantage during policy training.

Algorithms tested -

1. Vanilla Policy Gradient
2. Proximal Policy Optimization

After training, the algorithms are compared with a MaxTreatment, NoTreatment and Random policy.

## Training Rewards

### Vanilla Policy Gradient Training

![vpg_train_plot](./plots/vpg_train_rewards.png)

### Causal Policy Gradient Training

![causal_pg_train_plot](./plots/cpg_train_rewards.png)

### Proximal Policy Optimisation Training

![ppo_train_plot](./plots/ppo_train_rewards.png)

### Causal Proximal Policy Optimisation Training

![causal_ppo_train_plot](./plots/cppo_train_rewards.png)

### Combined Training Results

![all_train_plot](./plots/all_train_rewards.png)

## Causal Factor During Training

### Causal PG Causal Factor

![causal_pg_causal_factor_episodic_plot](./plots/cpg_episodic_causal_factor.png)
![causal_pg_causal_factor_mean_plot](./plots/cpg_mean_causal_factor.png)

### Causal PPO Causal Factor

![causal_ppo_causal_factor_episodic_plot](./plots/cppo_episodic_causal_factor.png)
![causal_ppo_causal_factor_mean_plot](./plots/cppo_mean_causal_factor.png)

### Combined Causal Factor Results

![all_causal_factor_episodic_plot](./plots/both_episodic_causal_factor.png)
![all_causal_factor_mean_plot](./plots/both_mean_causal_factor.png)

## Trained Behavior Evaluation

### Vanilla Policy Gradient Evaluation

![vpg_behavior_alone_plot](./plots/vpg_alone_behavior_120720193150.png)
![vpg_behavior_plot](./plots/vpg_behavior_120720193152.png)

### Causal Policy Gradient Evaluation

![causal_pg_behavior_alone_plot](./plots/causal_pg_alone_behavior_120720193155.png)
![causal_pg_behavior_plot](./plots/causal_ppo_behavior_120720193204.png)

### Proximal Policy Optimisation Evaluation

![ppo_behavior_alone_plot](./plots/ppo_behavior_120720193200.png)
![ppo_behavior_plot](./plots/ppo_alone_behavior_120720193158.png)

### Causal Proximal Policy Optimisation Evaluation

![causal_ppo_behavior_alone_plot](./plots/causal_ppo_alone_behavior_120720193202.png)
![causal_ppo_behavior_plot](./plots/causal_ppo_behavior_120720193204.png)

### Combined Evaluation Results

![all_behavior_plot](./plots/all_behavior_120720193207.png)
