import numpy as np
import torch
import dill
import math


class QLearning:
    def __init__(self, state_dim, act_dim, config, mode='train'):
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.gamma = config.gamma
        self.lr = config.lr
        self.eps = 0
        self.sample_cnt = 0
        #考虑eps衰减
        self.eps_start = config.eps_start
        self.eps_end = config.eps_end
        self.eps_decay = config.eps_decay
        self.mode = mode
        #given state_dim，当然用default dict也可以
        self.Q_table = np.zeros((state_dim, act_dim))

    def choose_action(self, state):
        if self.mode == 'train':
            return self.explore_policy(state)
        elif self.mode == 'eval':
            return self.target_policy(state)

    def explore_policy(self, state):
        self.sample_cnt += 1
        self.eps = self.eps_end + (self.eps_start-self.eps_end) * math.exp(-1.*self.sample_cnt/self.eps_decay)

        if np.random.random() < self.eps:
            return np.random.choice(self.act_dim)
        else:
            max_qsa = np.max(self.Q_table[state])
            return np.random.choice(np.where(self.Q_table[state] == max_qsa)[0])
        #return np.random.choice(self.act_dim)


    def target_policy(self, state):
        max_qsa = np.max(self.Q_table[state])
        return np.random.choice(np.where(self.Q_table[state] == max_qsa)[0])

    def update(self, state, action, reward, nxt_state, done):
        if done:
            td_error = reward - self.Q_table[state][action]
        else:
            #nxt_action = self.target_policy(nxt_state)
            td_error = reward + self.gamma*np.max(self.Q_table[nxt_state]) - self.Q_table[state][action]
        self.Q_table[state][action] += self.lr*td_error

    def save(self, path):
        """将Q_table中的数据保存到对应文件中"""
        torch.save(
            obj=self.Q_table,
            f=path + "Q_table.npy",
            pickle_module=dill
        )

    def load(self, path):
        self.Q_table = torch.load(f=path + "Q_table.npy", pickle_module=dill)