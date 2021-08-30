import numpy as np
import torch
import torch.nn as nn

from common.model import MLP
from common.model import MLP_Categorical


class PGTrainer:
    def __init__(self, state_dim, act_dim, cfg):
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.device = cfg.device
        self.clip_gradient = cfg.clip_gradient
        self.gamma = cfg.gamma
        self.hidden_dims = cfg.hidden_dims
        self.policy = MLP_Categorical(self.state_dim, self.act_dim, self.hidden_dims).to(self.device)
        self.optimizer = torch.optim.RMSprop(self.policy.parameters(), lr=cfg.lr)

    def to_tensor(self, array):
        return torch.from_numpy(array).to(self.device)

    def to_array(self, tensor):
        return tensor.cpu().detach().numpy()

    def choose_action(self, state):
        assert isinstance(state, np.ndarray)
        assert state.ndim == 1

        state = self.to_tensor(state)
        action = self.policy(state).item()
        return action

    def compute_log_probs(self, states, actions):
        if states.ndim == 1:
            states = self.to_tensor(states[np.newaxis, :])
        else:
            states = self.to_tensor(states)

        actions = self.to_tensor(actions)
        log_probs = self.policy.log_prob(states, actions)
        assert log_probs.dim() == 1, log_probs.dim()
        return log_probs

    def update(self, states_list, actions_list, rewards_list):
        values = self.compute_values(rewards_list)
        states = np.concatenate(states_list)
        actions = np.concatenate(actions_list)
        #normalization
        #values = (values-np.mean(values))/np.maximum(np.std(values), 1e-6)
        self.update_policy(states, actions, values)

    def compute_values(self, rewards_list):
        values = []
        for rewards in rewards_list:
            returns = np.zeros_like(rewards, np.float32)
            Q = 0
            n = len(rewards)
            for i, reward in enumerate(reversed(rewards)):
                Q = self.gamma * Q + reward
                returns[n - i - 1] = Q
            values.append(returns)
        values = np.concatenate(values)
        return values

    def update_policy(self, states, actions, advantages):
        #type of parameters should be np.ndarray
        advantages = self.to_tensor(advantages)
        self.policy.train()
        log_probs = self.compute_log_probs(states, actions)

        assert log_probs.shape == advantages.shape

        loss = (-advantages*log_probs).mean()
        self.optimizer.zero_grad()
        loss.backward()
        if self.clip_gradient is not None:
            torch.nn.utils.clip_grad_norm_(self.policy.parameters, self.clip_gradient)
        self.optimizer.step()
        self.policy.eval()

    def save(self, path):
        torch.save(self.policy.state_dict(), path + 'pg.pth')

    def load(self, path):
        self.policy.load_state_dict(torch.load(path + 'pg.pth'))


class PGBaselineTrainer(PGTrainer):
    def __init__(self, state_dim, act_dim, cfg):
        super(PGBaselineTrainer, self).__init__(state_dim, act_dim, cfg)
        '''self.state_dim = state_dim
        self.act_dim = act_dim
        self.device = cfg.device
        self.clip_gradient = cfg.clip_gradient
        self.gamma = cfg.gamma
        self.hidden_dims = cfg.hidden_dims
        self.policy = MLP_Categorical(self.state_dim, self.act_dim, self.hidden_dims).to(self.device)
        self.optimizer = torch.optim.RMSprop(self.policy.parameters(), lr=cfg.lr)'''
        self.baseline = MLP(self.state_dim, 1, self.hidden_dims).to(self.device)
        self.baselineloss = nn.MSELoss()
        self.baselineoptimizer = torch.optim.RMSprop(self.baseline.parameters(), lr=cfg.lr)

    def update(self, states_list, actions_list, rewards_list):

        values = self.compute_values(rewards_list)
        states = np.concatenate(states_list)
        actions = np.concatenate(actions_list)
        #normalization
        #values = (values - np.mean(values)) / np.maximum(np.std(values), 1e-6)

        baselines = self.baseline(self.to_tensor(states)).flatten()
        baselines = self.to_array(baselines)
        assert baselines.shape == values.shape
        #baselines = (baselines - np.mean(baselines) + np.mean(values)) / np.maximum(np.std(baselines), 1e-6) * np.maximum(np.std(values), 1e-6)
        advantages = values - baselines

        self.update_baseline(values, states)
        self.update_policy(states, actions, advantages)

    def update_baseline(self, values, states):
        self.baseline.train()
        values = self.to_tensor(values[:, np.newaxis])
        baselines = self.baseline(self.to_tensor(states))

        loss = self.baselineloss(input=baselines, target=values)
        self.baselineoptimizer.zero_grad()
        loss.backward()
        if self.clip_gradient is not None:
            torch.nn.utils.clip_grad_norm_(self.policy.parameters, self.clip_gradient)
        self.baselineoptimizer.step()
        self.baseline.eval()

    def save(self, path):
        torch.save(self.policy.state_dict(), path + 'pg.pth')

    def load(self, path):
        self.policy.load_state_dict(torch.load(path + 'pg.pth'))