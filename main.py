import tianshou as ts
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import DummyVectorEnv
from tianshou.trainer import OnpolicyTrainer
from tianshou.policy import PPOPolicy
from tianshou.utils.net.common import Net
from tianshou.utils.net.continuous import ActorProb, Critic

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torch.distributions import Independent, Normal
import numpy as np
import argparse

from env import GymReachingEnvironment
print(f'tianshou version: {ts.__version__}')

sim_parser = argparse.ArgumentParser()
sim_parser.add_argument('--arm', type=str, default='right', help='Which arm to use')
sim_parser.add_argument('--device', type=str, choices=['cuda', 'cpu'], default='cpu',
                        help='Device to run the model on (cuda or cpu)')
sim_args = sim_parser.parse_args()
device = torch.device(sim_args.device if torch.cuda.is_available() and sim_args.device == 'cuda' else 'cpu')

print(f'device: {device}')

# Create a vector of environments
train_envs = DummyVectorEnv([lambda: GymReachingEnvironment() for _ in range(10)])
test_envs = DummyVectorEnv([lambda: GymReachingEnvironment() for _ in range(3)])

# Set up the state shape and action shape
state_shape = train_envs.observation_space[0].shape[0]
action_shape = train_envs.action_space[0].shape[0]

# Create actor and critic networks
net_a = Net(state_shape, hidden_sizes=(64, 64), device=device)
actor = ActorProb(net_a, action_shape, device=device).to(device)
net_c = Net(state_shape, hidden_sizes=(64, 64), device=device)
critic = Critic(net_c, device=device).to(device)

# Orthogonal initialization
for m in list(actor.modules()) + list(critic.modules()):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.orthogonal_(m.weight)
        torch.nn.init.zeros_(m.bias)

# Create the PPO policy
optim = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=3e-4)

def dist(logits):
    mean, log_std = logits
    return Independent(Normal(mean, log_std.exp()), 1)


# Create the PPO policy
policy = PPOPolicy(
    actor=actor,
    critic=critic,
    optim=optim,
    dist_fn=dist,
    action_space=train_envs.action_space[0],
    action_bound_method=None,  # Changed from "clip" to None
    action_scaling=False,  # Disabled action scaling
    discount_factor=0.99,
    max_grad_norm=0.5,
    gae_lambda=0.95,
    vf_coef=0.5,
    ent_coef=0.01,
    reward_normalization=False,
    dual_clip=None,
    value_clip=False,
    deterministic_eval=False,
    advantage_normalization=True,
    recompute_advantage=False
)

# Set up the replay buffer
buffer = VectorReplayBuffer(20000, len(train_envs))

# Create collectors for training and testing
train_collector = Collector(policy, train_envs, buffer)
test_collector = Collector(policy, test_envs)


# Training function
def train_ppo(
    max_epoch: int,
    step_per_epoch: int,
    repeat_per_collect: int,
    batch_size: int,
    step_per_collect: int
):
    trainer = OnpolicyTrainer(
        policy=policy,
        train_collector=train_collector,
        test_collector=test_collector,
        max_epoch=max_epoch,
        step_per_epoch=step_per_epoch,
        step_per_collect=step_per_collect,
        repeat_per_collect=repeat_per_collect,
        episode_per_test=5,
        batch_size=batch_size,
        save_best_fn=lambda policy: torch.save(policy.state_dict(), 'ppo_policy.pth'),
        logger=ts.utils.TensorboardLogger(SummaryWriter('log/ppo'))
    )

    # Start training
    result = trainer.run()

    # Save the best policy
    torch.save(policy.state_dict(), 'ppo_policy.pth')

    return result


# Run the training
result = train_ppo(
    max_epoch=1000,
    step_per_epoch=20000,
    repeat_per_collect=10,
    batch_size=64,
    step_per_collect=2000
)

# Print the results
print(f'Finished training! Used {result.duration}')
print(f'Best reward: {result.best_reward}')
