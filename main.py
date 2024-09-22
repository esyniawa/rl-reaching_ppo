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
import matplotlib.pyplot as plt
from os import path, makedirs
import argparse

from env import GymReachingEnvironment
print(f'tianshou version: {ts.__version__}')


# Training function
def train_policy(
        policy,
        max_epoch: int,
        step_per_epoch: int,
        repeat_per_collect: int | None,  # set to None for off-policy algorithms
        batch_size: int,
        step_per_collect: int
):
    # initialize trainer with policy
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
        logger=ts.utils.TensorboardLogger(SummaryWriter(f'log/{policy.__class__.__name__}')),
    )

    # Start training
    result = trainer.run()

    # Save the best policy
    folder = 'results/best_policies/'
    if not path.exists(folder):
        makedirs(folder)
    torch.save(policy.state_dict(), f'{policy.__class__.__name__}_policy.pth')

    return result, policy


def evaluate_policy(policy,
                    save_folder: str,
                    num_trials: int = 10,
                    max_steps: int = 500,
                    track_error_per_trial: bool = False):

    env = GymReachingEnvironment()
    total_reward = 0
    total_steps = 0
    successes = 0
    errors = []

    for trial in range(num_trials):
        obs, _ = env.reset()
        episode_reward = 0

        # track error per trial if needed
        if track_error_per_trial:
            error = []

        for _ in range(max_steps):
            action = policy.forward(obs)[0]
            obs, reward, done, trunc, info = env.step(action)
            episode_reward += reward

            if track_error_per_trial:
                error.append(info['error'])

            # break trial when target is reached
            if done:
                if info["success"]:
                    successes += 1
                total_steps += info["num_steps"]
                break

        total_reward += episode_reward
        errors.append(info['error'])

        if track_error_per_trial:
            save_folder = f'results/{save_folder}/'
            if not path.exists(save_folder):
                makedirs(save_folder)
            error = np.array(error)
            np.save(save_folder + f'{policy.__class__.__name__}_error_{trial}.npy', error)

            # plot error
            fig, ax = plt.subplots()
            ax.plot(error)
            ax.set_xlabel('Step')
            ax.set_ylabel('Error')
            ax.set_title(f'{policy.__class__.__name__} Error')
            plt.savefig(save_folder + f'{policy.__class__.__name__}_error_{trial}.pdf')
            plt.close(fig)

    # compute average
    avg_reward = total_reward / num_trials
    avg_steps = total_steps / num_trials
    success_rate = successes / num_trials

    return avg_reward, avg_steps, success_rate, errors


if __name__ == '__main__':
    sim_parser = argparse.ArgumentParser()
    sim_parser.add_argument('--arm', type=str, default='right', help='Which arm to use')
    sim_parser.add_argument('--device', type=str, choices=['cuda', 'cpu'], default='cpu',
                            help='Device to run the model on (cuda or cpu)')
    sim_parser.add_argument('--num_training_trials', type=int, default=1_000,)
    sim_parser.add_argument('--num_episodes', type=int, default=10_000,)
    sim_parser.add_argument('--hidden_layer_size', type=int, default=64,)
    sim_parser.add_argument('--plot_test_errors', type=bool, default=True,)
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
    net_a = Net(state_shape, hidden_sizes=(sim_args.hidden_layer_size, sim_args.hidden_layer_size), device=device)
    actor = ActorProb(net_a, action_shape, device=device).to(device)
    net_c = Net(state_shape, hidden_sizes=(sim_args.hidden_layer_size, sim_args.hidden_layer_size), device=device)
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
    buffer = VectorReplayBuffer(sim_args.num_episodes, len(train_envs))

    # Create collectors for training and testing
    train_collector = Collector(policy, train_envs, buffer)
    test_collector = Collector(policy, test_envs)

    # Run the training
    result, policy = train_policy(
        policy=policy,
        max_epoch=sim_args.num_training_trials,
        step_per_epoch=sim_args.num_episodes,
        repeat_per_collect=10,
        batch_size=64,
        step_per_collect=sim_args.num_episodes // 10  # divide through the total number of "workers"
    )

    # test policy
    avg_reward, avg_steps, success_rate, errors = evaluate_policy(
        policy=policy,
        save_folder='test_runs/',
        num_trials=50,
        track_error_per_trial=True
    )

    # plot errors
    if sim_args.plot_test_errors:
        fig = plt.figure()
        plt.plot(errors)
        plt.xlabel('Trial')
        plt.ylabel('Error')
        plt.title(f'Errors of {policy.__class__.__name__} on {sim_args.arm} arm')
        if not path.exists('results/'):
            makedirs('results/')
        plt.savefig(f'results/{policy.__class__.__name__}_error_per_trial.pdf')

    # Print the results
    print(f'Finished training! Used {result.duration}')
    print(f'Best reward: {result.best_reward}')
