import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from kinematics.planar_arm import PlanarArm
from utils import gaussian_reward


def generate_random_target(
        arm: str,
        init_thetas: np.ndarray,
        min_distance: float = 50.,
        clip_borders_theta: float = np.radians(5.),
        normalize_xy: bool = False
) -> tuple[np.ndarray, np.ndarray]:

    valid = False

    while not valid:
        random_thetas = PlanarArm.random_theta(clip_borders_lower=clip_borders_theta,
                                               clip_borders_upper=clip_borders_theta)

        random_xy = PlanarArm.forward_kinematics(arm=arm,
                                                 thetas=random_thetas,
                                                 radians=True)[:, -1]

        init_xy = PlanarArm.forward_kinematics(arm=arm,
                                               thetas=init_thetas,
                                               radians=False)[:, -1]
        distance = np.linalg.norm(init_xy - random_xy)

        if distance > min_distance:
            valid = True

    if not normalize_xy:
        return random_thetas, random_xy
    else:
        # Normalize random_xy
        normalized_x = (random_xy[0] - PlanarArm.x_limits[0]) / (PlanarArm.x_limits[1] - PlanarArm.x_limits[0])
        normalized_y = (random_xy[1] - PlanarArm.y_limits[0]) / (PlanarArm.y_limits[1] - PlanarArm.y_limits[0])
        normalized_xy = np.array([normalized_x, normalized_y])

        return random_thetas, normalized_xy


class ReachingEnvironment:
    def __init__(self,
                 arm: str = 'right',
                 init_thetas: np.ndarray = np.radians((90, 90)),
                 radians: bool = True):

        if not radians:
            self.init_thetas = np.radians(init_thetas)
        else:
            self.init_thetas = init_thetas

        self.arm = arm
        self.current_thetas = self.init_thetas.copy()
        self.current_pos = PlanarArm.forward_kinematics(self.arm, self.current_thetas, radians=True)[:, -1]

        self.target_thetas, self.target_pos = None, None
        # modulates the reward function
        self.max_distance = 50.

    def new_target(self, clip_borders_theta: float = np.radians(0.0)):
        self.init_thetas = self.current_thetas.copy()
        self.target_thetas, self.target_pos = generate_random_target(arm=self.arm,
                                                                     init_thetas=self.init_thetas,
                                                                     normalize_xy=True,
                                                                     clip_borders_theta=clip_borders_theta)

    def update_position(self):
        self.current_pos = PlanarArm.forward_kinematics(self.arm, self.current_thetas, radians=True, check_limits=False)[:, -1]
        return self.current_pos

    def reset(self):
        self.current_thetas = self.init_thetas.copy()
        self.update_position()
        return np.concatenate([self.current_thetas, self.target_pos])

    def step(self,
             action: np.ndarray,
             max_angle_change: float = np.radians(10),
             clip_thetas: bool = True):

        delta_thetas = np.clip(action, -max_angle_change, max_angle_change)  # prevent very large angle changes

        # clip angles to joint constraints
        new_thetas = self.current_thetas + delta_thetas
        if clip_thetas:
            new_thetas = PlanarArm.clip_values(new_thetas, radians=True)

        # Calculate new position
        new_pos = PlanarArm.forward_kinematics(self.arm, new_thetas, radians=True, check_limits=False)[:, -1]

        # Calculate reward
        # TODO: Find better reward function
        distance = np.linalg.norm(new_pos - self.target_pos)
        reward = gaussian_reward(distance, sigma=self.max_distance)
        # reward smoother actions
        reward += 0.1 * (1.0 - np.sum(np.abs(action)) / (2 * max_angle_change))
        done = distance < 10  # 10mm threshold

        self.current_thetas = new_thetas
        self.update_position()
        return np.concatenate([new_thetas, self.target_pos]), reward, done


class GymReachingEnvironment(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self,
                 arm='right',
                 init_thetas=np.radians((90, 90)),
                 radians=True):
        super().__init__()

        self.env = ReachingEnvironment(arm, init_thetas, radians)

        # Define action and observation space
        self.action_space = Box(low=-np.radians(10), high=np.radians(10), shape=(2,), dtype=np.float32)
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)

        self.num_steps = 0
        self.reward_threshold = 0.95  # Define a reward threshold for "solving" the environment

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # set new target
        self.env.new_target()
        self.num_steps = 0

        observation = self.env.reset()

        # Create an info dict (empty for now, but you can add relevant info here)
        info = {
            'num_steps': self.num_steps,
            'error': np.linalg.norm(self.env.target_pos - self.env.current_pos),
            'target_pos': self.env.target_pos,
            'current_pos': self.env.current_pos,
            'success': False
        }

        return observation.astype(np.float32), info

    def step(self, action):
        self.num_steps += 1
        observation, reward, done = self.env.step(action, max_angle_change=np.radians(10))

        # In Gymnasium, we need to include a "truncated" boolean
        # For simplicity, we'll set it to False always
        truncated = False

        # Create an info dict (empty for now, but you can add relevant info here)
        info = {
            'num_steps': self.num_steps,
            'error': np.linalg.norm(self.env.target_pos - self.env.current_pos),
            'target_pos': self.env.target_pos,
            'current_pos': self.env.current_pos,
            'success': True if done else False
        }

        return observation.astype(np.float32), reward, done, truncated, info

    def render(self):
        if not hasattr(self, 'fig'):
            self.fig, self.ax = plt.subplots(figsize=(8, 8))
            self.ax.set_xlim(PlanarArm.x_limits)
            self.ax.set_ylim(PlanarArm.y_limits)
            self.ax.set_aspect('equal')
            self.ax.grid(True)
            self.line, = self.ax.plot([], [], 'o-', lw=2)
            self.target, = self.ax.plot([], [], 'r*', markersize=10)

        # Get current arm position
        arm_positions = PlanarArm.forward_kinematics(self.env.arm, self.env.current_thetas, radians=True, check_limits=False)

        # Update arm line
        x_data = arm_positions[0].tolist()
        y_data = arm_positions[1].tolist()
        self.line.set_data(x_data, y_data)

        # Update target position
        target_x = self.env.target_pos[0] * (PlanarArm.x_limits[1] - PlanarArm.x_limits[0]) + PlanarArm.x_limits[0]
        target_y = self.env.target_pos[1] * (PlanarArm.y_limits[1] - PlanarArm.y_limits[0]) + PlanarArm.y_limits[0]
        self.target.set_data([target_x], [target_y])

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.05)

    def close(self):
        if hasattr(self, 'fig'):
            plt.close(self.fig)
            del self.fig


def create_animation(env: GymReachingEnvironment, policy, max_steps: int = 500, filename: str = 'reaching_animation.mp4'):
    obs, _ = env.reset()

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(PlanarArm.x_limits)
    ax.set_ylim(PlanarArm.y_limits)
    ax.set_aspect('equal')
    ax.grid(True)
    line, = ax.plot([], [], 'o-', lw=2)
    target, = ax.plot([], [], 'r*', markersize=10)

    def animate(i):
        if i > 0:
            nonlocal obs
            action = policy.forward(obs)[0]
            obs, _, done, _, _ = env.step(action)
            if done:
                return line, target

        arm_positions = PlanarArm.forward_kinematics(env.env.arm, env.env.current_thetas, radians=True)
        x_data = arm_positions[0].tolist()
        y_data = arm_positions[1].tolist()
        line.set_data(x_data, y_data)

        target_x = env.env.target_pos[0] * (PlanarArm.x_limits[1] - PlanarArm.x_limits[0]) + PlanarArm.x_limits[0]
        target_y = env.env.target_pos[1] * (PlanarArm.y_limits[1] - PlanarArm.y_limits[0]) + PlanarArm.y_limits[0]
        target.set_data([target_x], [target_y])

        return line, target

    anim = FuncAnimation(fig, animate, frames=max_steps, interval=50, blit=True)
    anim.save(filename, writer='ffmpeg', fps=30)
    plt.close(fig)


if __name__ == '__main__':
    env = GymReachingEnvironment()
    print(env.observation_space)
    state, _ = env.reset()
    env.render()
    for _ in range(100):
        action = np.random.uniform(low=-np.radians(10), high=np.radians(10), size=(2,))
        state, reward, done, trunc, info = env.step(action)
        env.render()
        print(reward, done, info['success'])
        if done:
            break

    env.close()
