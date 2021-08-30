import numpy as np
import torch
import torch.nn as nn

from common.model import MLP
from common.model import MLP_Categorical
from common.memory import ReplayBuffer


class PPOAgent:
    def __init__(self, state_dim, act_dim, cfg):
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.device = cfg.device
        self.gamma = cfg.gamma
        self.batch_size = cfg.batch_size
        self.hidden_dim = cfg.hidden_dim
        self.gae_lambda = cfg.gae_lambda  #for gae
        self.clip_param = cfg.clip_param
        self.n_epoch = cfg.n_epoch

        #should include (state, action, log_prob, gae, returns=gae+value_pred,)
        self.memory = ReplayBuffer(cfg.capacity)

        self.actor = MLP_Categorical(state_dim, act_dim, self.hidden_dim).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=cfg.lr)

        self.critic = MLP(state_dim, 1, self.hidden_dim).to(self.device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=cfg.lr)
        self.critic_loss = nn.MSELoss()

    def to_tensor(self, array):
        return torch.from_numpy(array).to(self.device)

    def to_array(self, tensor):
        return tensor.cpu().detach().numpy()

    def choose_action(self, state):
        assert isinstance(state, np.ndarray)
        assert state.ndim == 1

        tensor_state = self.to_tensor(state)
        return self.actor(tensor_state).item()

    def compute_log_probs(self, states, actions):
        if isinstance(states, torch.Tensor):
            log_probs = self.actor.log_prob(states, actions)
            assert log_probs.dim() == 1, log_probs.dim()
            return log_probs

        if states.ndim == 1:
            states = self.to_tensor(states[np.newaxis, :])
        else:
            states = self.to_tensor(states)

        actions = self.to_tensor(actions)
        log_probs = self.actor.log_prob(states, actions)
        assert log_probs.dim() == 1, log_probs.dim()
        return log_probs

    def compute_gae(self, states, next_states, rewards, masks):
        tensor_states = self.to_tensor(states)
        tensor_next_states = self.to_tensor(next_states)

        values = self.to_array(self.critic(tensor_states).detach().squeeze())
        next_values = self.to_array(self.critic(tensor_next_states).detach().squeeze())

        returns = np.zeros_like(rewards)
        gaes = np.zeros_like(rewards)
        gae = 0
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + self.gamma * (1-masks[step]) * next_values[step] - values[step]
            gae = self.gamma * self.gae_lambda * gae + delta
            gaes[step] = gae
            returns[step] = gae + values[step]
        return gaes, returns

    def process_samples(self, states, actions, rewards):
        #一次epsiode结束后process samples， 将这些放入memory。
        #若干次后，通过从memory中sample出进行更新
        N = len(states)

        next_states = states[1:]
        next_states.append(np.zeros_like(states[0]))
        #print(states[1], next_states[0])
        #assert 0
        masks = [False] * (N-1) + [True]

        states = np.array(states, dtype=np.float32)
        actions = np.array(actions, dtype=np.float32)
        rewards = np.array(rewards, dtype=np.float32)
        next_states = np.array(next_states, dtype=np.float32)
        masks = np.array(masks, dtype=np.float32)

        log_probs = self.compute_log_probs(states, actions).detach()    #作为old_log_prob，不需要保留梯度
        gaes, returns = self.compute_gae(states, next_states, rewards, masks)
        return log_probs, gaes, returns

    def update(self):
        for _ in range(2*self.n_epoch):
            batch = self.memory.sample(self.batch_size)
            state_batch = self.to_tensor(
                np.stack(transition[0] for transition in batch)
            )
            action_batch = self.to_tensor(
                np.stack(transition[1] for transition in batch)
            )
            old_log_prob_batch = self.to_tensor(
                np.stack(transition[2] for transition in batch)
            )
            gae_batch = self.to_tensor(
                np.stack(transition[3] for transition in batch)
            )
            return_batch = self.to_tensor(
                np.stack(transition[4] for transition in batch)
            )

            #update critic
            self.critic.train()
            values = self.critic(state_batch).squeeze().to(self.device)
            critic_loss = self.critic_loss(input=values, target=return_batch)
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()
            self.critic.eval()

            #update actor
            self.actor.train()
            new_log_prob = self.compute_log_probs(state_batch, action_batch)
            ratio = torch.exp(new_log_prob - old_log_prob_batch)
            surr1 = ratio * gae_batch
            surr2 = torch.clamp(ratio, 1-self.clip_param, 1+self.clip_param) * gae_batch
            actor_loss = -torch.min(surr1, surr2).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            self.actor.eval()

    def save(self, path):
        torch.save(self.actor.state_dict(), path + 'ppo_actor.pth')
        torch.save(self.critic.state_dict(), path + 'ppo_critic.pth')

    def load(self, path):
        self.actor.load_state_dict(torch.load(path + 'ppo_actor.pth'))
        self.critic.load_state_dict(torch.load(path + 'ppo_critic.pth'))



