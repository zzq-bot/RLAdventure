import numpy as np
import torch
import dill
from collections import defaultdict


class FirstVisitMC:

    def __init__(self, act_dim, config):
        self.act_dim = act_dim
        self.gamma = config.gamma
        self.eps = config.eps

        #环境有无限的state
        self.Q_table = defaultdict(lambda: np.zeros(self.act_dim))
        self.N = defaultdict(float)

    def choose_action(self, state):
        if state in self.Q_table.keys():
            if np.random.random() < self.eps:
                return np.random.choice(self.act_dim)
            else:
                max_qsa = np.max(self.Q_table[state])
                return np.random.choice(np.where(self.Q_table[state] == max_qsa)[0])
        else:
            return np.random.choice(self.act_dim)

    def update(self, one_ep_experience):
        #experience中应该每个元素是(s_t,a_t,r_t+1)
        occured_state_action_pair = set()
        length = len(one_ep_experience)
        G = 0  # return
        returns = [0]*length
        for i in reversed(range(length)):
            _, _, reward = one_ep_experience[i]
            G = self.gamma*G + reward
            returns[i] = G
        for i in range(length):
            state, act, _ = one_ep_experience[i]
            if (state, act) not in occured_state_action_pair:
                occured_state_action_pair.add((state, act))
                self.N[(state, act)] += 1.0
                self.Q_table[state][act] += (returns[i]-self.Q_table[state][act])/self.N[(state, act)]

    def save(self, path):
        """将Q_table中的数据保存到对应文件中"""
        torch.save(
            obj=self.Q_table,
            f=path+"Q_table",
            pickle_module=dill
        )

    def load(self, path):
        self.Q_table = torch.load(f=path+"Q_table", pickle_module=dill)
