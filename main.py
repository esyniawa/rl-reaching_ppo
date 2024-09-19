import tianshou as ts
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import DummyVectorEnv
from tianshou.policy import PPOPolicy
from tianshou.utils.net.common import Net
from tianshou.utils.net.continuous import ActorProb, Critic

import torch
from torch import nn
import numpy as np

from env import GymReachingEnvironment
print(f'tianshou version: {ts.__version__}')

# Create a vector of environments
train_envs = DummyVectorEnv([lambda: GymReachingEnvironment() for _ in range(10)])
test_envs = DummyVectorEnv([lambda: GymReachingEnvironment() for _ in range(3)])

# Set up the state shape and action shape
state_shape = train_envs.observation_space[0].shape
action_shape = train_envs.action_space[0].shape