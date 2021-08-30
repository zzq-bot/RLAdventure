import numpy as np
import torch
import torch.nn as nn

from common.model import DDPGActor, DDPGCritic
from common.memory import ReplayBuffer


class DDPGAgent:
    def __init__(self, state_dim, act_dim, cfg):
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.device = cfg.device

        self.critic = DDPGCritic(state_dim, act_dim, cfg.hidden_dim).to(self.device)
        self.actor = DDPGActor(state_dim, act_dim, cfg.hidden_dim).to(self.device)
        self.target_critic = DDPGCritic(state_dim, act_dim, cfg.hidden_dim).to(self.device)
        self.target_actor = DDPGActor(state_dim, act_dim, cfg.hidden_dim).to(self.device)

        #拷贝参数
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.target_critic.eval()
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_actor.eval()

        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=cfg.critic_lr)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)

        self.memory = ReplayBuffer(cfg.capacity)
        self.batch_size = cfg.batch_size
        self.soft_tau = cfg.soft_tau
        self.gamma = cfg.gamma

    def choose_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action = self.actor(state)
        return action.detach().cpu().numpy()[0, 0] #[[act]]，在该环境下，act是一维连续的
    
    def update(self):
        if len(self.memory) < self.batch_size:
            return

        state, action, reward, next_state, done = zip(*self.memory.sample(self.batch_size))
        state = torch.FloatTensor(state).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        action = torch.FloatTensor(action).to(self.device)  #(np.array([]), np.array([])....)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(self.device)
        done = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(self.device)
        #reward, done转成对应列向量
        #print(state.shape, next_state.shape, action.shape, reward.shape, done.shape)
        self.actor.train()
        self.critic.train()
        self.target_actor.train()
        self.target_critic.train()

        policy_loss = self.critic(state, self.actor(state))
        policy_loss = -policy_loss.mean()
        next_action = self.target_actor(next_state)
        target_value = self.critic(next_state, next_action)
        expected_value = reward + (1. - done) * self.gamma * target_value
        expected_value = torch.clamp(expected_value, -np.inf, np.inf)
        value = self.critic(state, action)
        value_loss = nn.MSELoss()(input=value, target=expected_value.detach())

        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

        self.critic_optimizer.zero_grad()
        value_loss.backward()
        self.critic_optimizer.step()
        #更新target_network 参数
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.soft_tau) +
                param.data * self.soft_tau
            )
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.soft_tau) +
                param.data * self.soft_tau
            )

    def save(self, path):
        torch.save(self.actor.state_dict(), path + 'ddpgcheckpoint.pt')

    def load(self, path):
        self.actor.load_state_dict(torch.load(path + 'ddpgcheckpoint.pt'))
