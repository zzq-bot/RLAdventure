import numpy as np
import torch
import torch.nn as nn
import os

from common.model import MLP
from common.model import MLP_Categorical
from common.memory import PPOMemory


class PPOAgent:
    def __init__(self, state_dim, act_dim, cfg):
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.gamma = cfg.gamma
        self.policy_clip = cfg.policy_clip
        self.n_epochs = cfg.n_epochs
        self.gae_lambda = cfg.gae_lambda
        self.device = cfg.device

        self.actor = MLP_Categorical(state_dim, act_dim, cfg.hidden_dim).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=cfg.lr)

        self.critic = MLP(state_dim, 1, cfg.hidden_dim).to(self.device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=cfg.lr)

        self.memory = PPOMemory(cfg.batch_size)

    def choose_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action = self.actor(state)
        value = self.critic(state)
        prob = self.actor.log_prob(state, action)
        return action.item(), prob.item(), value.item()

    def update(self):
        state_arr, action_arr, old_prob_arr, vals_arr, reward_arr,\
            dones_arr, _ = self.memory.sample()
        values = vals_arr
        #计算gae和returns
        gaes = np.zeros_like(reward_arr, dtype=np.float32)
        returns = np.zeros_like(reward_arr, dtype=np.float32)
        n = len(reward_arr)
        gae = 0
        for step in reversed(range(len(reward_arr))):
            if step == n-1:
                delta = reward_arr[step] - values[step]
            else:
                delta = reward_arr[step] + (1-dones_arr[step]) * self.gamma * values[step+1] - values[step]
            gae = self.gamma * self.gae_lambda * gae + delta
            gaes[step] = gae
            returns[step] = gae + values[step]
        gaes = torch.FloatTensor(gaes).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)
        """advantage = np.zeros(len(reward_arr), dtype=np.float32)
        returns = np.zeros_like(reward_arr, dtype=np.float32)
        for t in range(len(reward_arr) - 1):
            discount = 1
            a_t = 0
            for k in range(t, len(reward_arr) - 1):
                a_t += discount * (reward_arr[k] + self.gamma * values[k + 1] * \
                                   (1 - int(dones_arr[k])) - values[k])
                discount *= self.gamma * self.gae_lambda
            advantage[t] = a_t
            returns[t] = values[t] + advantage[t]
        advantage = torch.tensor(advantage).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)"""
        for _ in range(self.n_epochs):
            state_arr, action_arr, old_prob_arr, vals_arr, reward_arr, \
                dones_arr, batches = self.memory.sample()

            for batch in batches:
                #将一个轨迹中的数据拆分成若干个batch
                states = torch.FloatTensor(state_arr[batch]).to(self.device)
                old_probs = torch.FloatTensor(old_prob_arr[batch]).to(self.device)
                actions = torch.FloatTensor(action_arr[batch]).to(self.device)
                new_probs = self.actor.log_prob(states, actions).to(self.device)
                critic_value = self.critic(states).squeeze().to(self.device)

                ratio = torch.exp(new_probs - old_probs)
                surr1 = ratio * gaes[batch]
                surr2 = torch.clamp(ratio, 1 - self.policy_clip, 1 + self.policy_clip) * gaes[batch]

                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = ((returns[batch]-critic_value)**2).mean()

                total_loss = actor_loss + 0.5*critic_loss
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                total_loss.backward()
                self.actor_optimizer.step()
                self.critic_optimizer.step()

        self.memory.clear()

    def save(self, path):
        actor_checkpoint = os.path.join(path, 'ppo_actor.pt')
        critic_checkpoint = os.path.join(path, 'ppo_critic.pt')
        torch.save(self.actor.state_dict(), actor_checkpoint)
        torch.save(self.critic.state_dict(), critic_checkpoint)

    def load(self, path):
        actor_checkpoint = os.path.join(path, 'ppo_actor.pt')
        critic_checkpoint = os.path.join(path, 'ppo_critic.pt')
        self.actor.load_state_dict(torch.load(actor_checkpoint))
        self.critic.load_state_dict(torch.load(critic_checkpoint))