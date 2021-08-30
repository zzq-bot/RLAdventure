import gym
import numpy as np


class NormalizedActions(gym.ActionWrapper):
    """
    将action规约在[-1,1]
    """
    def action(self, action):
        #将规约在[-1,1]之间的action恢复为正常范围的action
        low_bound = self.action_space.low
        upper_bound = self.action_space.high
        action = low_bound + (action + 1.) * 0.5 * (upper_bound - low_bound)
        action = np.clip(action, low_bound, upper_bound)
        return action

    def reverse_action(self, action):
        low_bound = self.action_space.low
        upper_bound = self.action_space.high
        action = 2*(action - low_bound) / (upper_bound - low_bound) - 1
        action = np.clip(action, -1, 1)
        return action


class OUNoise:
    """
    OU noise 探索
    """
    def __init__(self, action_space, mu=0.0, theta=0.15, max_sigma=0.3, min_sigma=0.3, decay_period=100000):
        self.mu = mu
        self.theta = theta
        self.sigma = max_sigma
        self.max_sigma = max_sigma
        self.min_sigma = min_sigma
        self.decay_period = decay_period
        self.action_dim = action_space.shape[0]
        self.low = action_space.low
        self.high = action_space.high
        self.reset()

    def reset(self):
        self.obs = np.ones(self.action_dim) * self.mu

    def evolve_obs(self):
        x = self.obs
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dim)
        self.obs = x + dx
        return self.obs

    def get_action(self, action, t=0):
        ou_obs = self.evolve_obs()
        self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(1.0, t / self.decay_period)
        return np.clip(action + ou_obs, self.low, self.high)
        #因为+ou_obs，action转为np.array([])类型
