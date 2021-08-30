import numpy as np
import torch
import torch.nn as nn
import math

from common.model import DuelingModel
from common.memory import ReplayBuffer


def to_tensor(x):
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x).type(torch.float32)
    assert isinstance(x, torch.Tensor)
    if x.dim() == 3 or x.dim() == 1:
        x = x.unsqueeze(0)
    assert x.dim() == 2 or x.dim() == 4
    return x


class DuelingDQNAgent:
    def __init__(self, state_dim, act_dim, cfg, mode='train'):
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.gamma = cfg.gamma
        self.device = cfg.device
        self.mode = mode
        self.frame_idx = 0  # 用于epsilon的衰减计数
        self.epsilon = lambda frame_idx: cfg.epsilon_end + (cfg.epsilon_start - cfg.epsilon_end) * math.exp(-1. * self.frame_idx / cfg.epsilon_decay)
        self.batch_size = cfg.batch_size

        self.memory = ReplayBuffer(cfg.capacity)
        self.policy_net = DuelingModel(state_dim, act_dim, cfg.hidden_dim).to(self.device)
        self.target_net = DuelingModel(state_dim, act_dim, cfg.hidden_dim).to(self.device)
        # 复制参数
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=cfg.lr)
        self.loss = nn.MSELoss()

    def process_state(self, state):
        return torch.from_numpy(state).type(torch.float32)

    def choose_action(self, state):
        processed_state = self.process_state(state)
        if self.mode == 'train':
            return self.explore(processed_state)
        else:
            return self.predict(processed_state)

    def explore(self, processed_state):
        self.frame_idx += 1
        if np.random.random() < self.epsilon(self.frame_idx):
            return np.random.choice(self.act_dim)
        else:
            return self.predict(processed_state)

    def predict(self, processed_state):
        with torch.no_grad():
            action_values = self.policy_net(processed_state).detach().numpy()
            max_qsa = np.max(action_values)
            return np.random.choice(np.where(action_values == max_qsa)[0])

    def update(self):
        assert len(self.memory) >= self.batch_size, len(self.memory)
        batch = self.memory.sample(self.batch_size)

        state_batch = to_tensor(
            np.stack(transition[0] for transition in batch)
        )
        action_batch = to_tensor(
            np.stack(transition[1] for transition in batch)
        )
        reward_batch = to_tensor(
            np.stack(transition[2] for transition in batch)
        )
        next_state_batch = to_tensor(
            np.stack(transition[3] for transition in batch)
        )
        done_batch = to_tensor(
            np.stack(transition[4] for transition in batch)
        )

        #1、计算Q_target
        with torch.no_grad():
            Q_t_plus_one = torch.max(self.target_net(next_state_batch), dim=1)[0]
            assert Q_t_plus_one.ndim == 1, Q_t_plus_one.shape
            assert Q_t_plus_one.shape[0] == self.batch_size
            Q_target = reward_batch + self.gamma * (1-done_batch) * Q_t_plus_one
            Q_target = Q_target.squeeze()
            assert Q_target.shape == (self.batch_size,)

        #2、SGD
        self.policy_net.train()
        Q_t = self.policy_net(state_batch).gather(dim=1, index=action_batch.long().view(-1, 1)).squeeze()
        assert Q_target.shape == Q_t.shape

        loss = self.loss(input=Q_t, target=Q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.policy_net.eval()

    def save(self, path):
        torch.save(self.policy_net.state_dict(), path + 'duelingdqn_checkpoint.pth')

    def load(self, path):
        self.policy_net.load_state_dict(torch.load(path + 'duelingdqn_checkpoint.pth'))
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
