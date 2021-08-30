import numpy as np
import torch
import dill
from collections import defaultdict


class Sarsa:
    def __init__(self, act_dim, sarsa_cfg):
        self.act_dim = act_dim
        self.gamma = sarsa_cfg.gamma
        self.eps = sarsa_cfg.eps
        self.lr = sarsa_cfg.lr

        self.Q_table = defaultdict(lambda: np.zeros(self.act_dim))

    def choose_action(self, state):
        if state in self.Q_table.keys():
            if np.random.random() < self.eps:
                return np.random.choice(self.act_dim)
            else:
                max_qsa = np.max(self.Q_table[state])
                return np.random.choice(np.where(self.Q_table[state] == max_qsa)[0])
        else:
            return np.random.choice(self.act_dim)

    def update(self, state, action, reward, nxt_state, nxt_action, done):
        if done:
            td_error = reward-self.Q_table[state][action]
        else:
            td_error = reward+self.gamma*self.Q_table[nxt_state][nxt_action]-self.Q_table[state][action]
        self.Q_table[state][action] += self.lr*td_error

    def save(self, path):
        """将Q_table中的数据保存到对应文件中"""
        torch.save(
            obj=self.Q_table,
            f=path + "sarsa_model.pkl",
            pickle_module=dill
        )

    def load(self, path):
        self.Q_table = torch.load(f=path + "sarsa_model.pkl", pickle_module=dill)
