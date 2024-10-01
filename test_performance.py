import argparse
import os
import pprint
import datetime

import gymnasium as gym
import numpy as np
import torch
from torch import nn
from torch.distributions import Distribution, Independent, Normal
from torch.optim.lr_scheduler import LambdaLR

from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import DummyVectorEnv
from tianshou.policy import PPOPolicy
from tianshou.trainer import OnpolicyTrainer
from tianshou.utils.net.common import Net
from tianshou.utils.net.continuous import ActorProb, Critic
from tianshou.utils import TensorboardLogger
from torch.utils.tensorboard import SummaryWriter


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="BipedalWalker-v3")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--buffer-size", type=int, default=4096)
    parser.add_argument("--hidden-sizes", type=int, nargs="*", default=[64, 64])
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument("--step-per-epoch", type=int, default=30000)
    parser.add_argument("--step-per-collect", type=int, default=2048)
    parser.add_argument("--repeat-per-collect", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--training-num", type=int, default=8)
    parser.add_argument("--test-num", type=int, default=10)
    parser.add_argument("--rew-norm", type=int, default=True)
    parser.add_argument("--vf-coef", type=float, default=0.25)
    parser.add_argument("--ent-coef", type=float, default=0.0)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--bound-action-method", type=str, default="clip")
    parser.add_argument("--lr-decay", type=int, default=True)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--eps-clip", type=float, default=0.2)
    parser.add_argument("--dual-clip", type=float, default=None)
    parser.add_argument("--value-clip", type=int, default=0)
    parser.add_argument("--norm-adv", type=int, default=0)
    parser.add_argument("--recompute-adv", type=int, default=1)
    parser.add_argument("--logdir", type=str, default="log")
    parser.add_argument("--render", type=float, default=0.0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resume-path", type=str, default=None)
    parser.add_argument("--watch", default=False, action="store_true")
    return parser.parse_args()


def make_env(task: str):
    return gym.make(task)


def test_ppo(args=get_args()):
    env = make_env(args.task)
    args.state_shape = env.observation_space.shape or env.observation_space.n
    args.action_shape = env.action_space.shape or env.action_space.n
    args.max_action = env.action_space.high[0]

    # Seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Model
    net_a = Net(args.state_shape, hidden_sizes=args.hidden_sizes, activation=nn.Tanh, device=args.device)
    actor = ActorProb(net_a, args.action_shape, unbounded=True, device=args.device).to(args.device)
    net_c = Net(args.state_shape, hidden_sizes=args.hidden_sizes, activation=nn.Tanh, device=args.device)
    critic = Critic(net_c, device=args.device).to(args.device)

    # Orthogonal initialization
    for m in list(actor.modules()) + list(critic.modules()):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            torch.nn.init.zeros_(m.bias)

    # Do last policy layer scaling
    for m in actor.mu.modules():
        if isinstance(m, torch.nn.Linear):
            m.weight.data.copy_(0.01 * m.weight.data)

    optim = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=args.lr)

    # PPO Policy
    def dist(loc, scale):
        return Independent(Normal(loc, scale), 1)

    policy = PPOPolicy(
        actor=actor,
        critic=critic,
        optim=optim,
        dist_fn=dist,
        discount_factor=args.gamma,
        gae_lambda=args.gae_lambda,
        max_grad_norm=args.max_grad_norm,
        vf_coef=args.vf_coef,
        ent_coef=args.ent_coef,
        reward_normalization=args.rew_norm,
        action_scaling=True,
        action_bound_method=args.bound_action_method,
        action_space=env.action_space,
        eps_clip=args.eps_clip,
        value_clip=args.value_clip,
        dual_clip=args.dual_clip,
        advantage_normalization=args.norm_adv,
        recompute_advantage=args.recompute_adv,
    )

    # Collectors
    train_envs = DummyVectorEnv([lambda: make_env(args.task) for _ in range(args.training_num)])
    test_envs = DummyVectorEnv([lambda: make_env(args.task) for _ in range(args.test_num)])
    train_collector = Collector(policy, train_envs, VectorReplayBuffer(args.buffer_size, len(train_envs)))
    test_collector = Collector(policy, test_envs)

    # Log
    log_path = os.path.join(args.logdir, args.task, 'ppo')
    writer = SummaryWriter(log_path)
    logger = TensorboardLogger(writer)

    def save_best_fn(policy):
        torch.save(policy.state_dict(), os.path.join(log_path, 'policy.pth'))

    def stop_fn(mean_rewards):
        return mean_rewards >= env.spec.reward_threshold if env.spec.reward_threshold else False

    # Training
    result = OnpolicyTrainer(
        policy=policy,
        train_collector=train_collector,
        test_collector=test_collector,
        max_epoch=args.epoch,
        step_per_epoch=args.step_per_epoch,
        repeat_per_collect=args.repeat_per_collect,
        episode_per_test=args.test_num,
        batch_size=args.batch_size,
        step_per_collect=args.step_per_collect,
        stop_fn=stop_fn,
        save_best_fn=save_best_fn,
        logger=logger,
        show_progress=True,
    ).run()

    pprint.pprint(result)

    # Watch the performance
    policy.eval()
    test_envs.seed(args.seed)
    test_collector.reset()
    result = test_collector.collect(n_episode=args.test_num, render=args.render)
    print(f'Final reward: {result["rews"].mean()}, length: {result["lens"].mean()}')


if __name__ == "__main__":
    test_ppo()
