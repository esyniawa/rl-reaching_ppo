import numpy as np


def gaussian_reward(error: float, sigma: float, amplitude: float = 1.):
    # error and sigma should be in mm
    return amplitude * np.exp(-error**2 / (2 * sigma**2))


def logarithmic_reward(current_pos, target_pos, max_distance):
    distance = np.linalg.norm(current_pos - target_pos)
    return 1 - np.log(1 + distance) / np.log(1 + max_distance)


def sigmoid_reward(current_pos, target_pos, max_distance, k=10):
    distance = np.linalg.norm(current_pos - target_pos)
    return 1 / (1 + np.exp(k * (distance / max_distance - 0.5)))
