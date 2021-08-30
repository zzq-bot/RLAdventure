import numpy as np
import torch
import torch.nn as nn

from common.model import MLP
from common.model import MLP_Categorical


class A2CTrainer_v0:
    def __init__(self, state_dim, act_dim, cfg):
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.device = cfg.device
        self.clip_gradient = cfg.clip_gradient
        self.gamma = cfg.gamma
        self.hidden_dims = cfg.hidden_dims
        self.num_critic_updates = cfg.num_critic_updates
        self.num_critic_update_steps = cfg.num_critic_update_steps

        self.actor = MLP_Categorical(state_dim, act_dim, self.hidden_dims).to(self.device)
        self.actor_optimizer = torch.optim.RMSprop(self.actor.parameters(), lr=cfg.lr)
        self.critic = MLP(state_dim, 1, self.hidden_dims).to(self.device)
        self.critic_loss = nn.MSELoss()
        self.critic_optimizer = torch.optim.RMSprop(self.critic.parameters(), lr=cfg.lr)

    def to_tensor(self, array):
        return torch.from_numpy(array).to(self.device)

    def to_array(self, tensor):
        return tensor.cpu().detach().numpy()

    def choose_action(self, state):
        assert isinstance(state, np.ndarray)
        assert state.ndim == 1

        tensor_state = self.to_tensor(state)
        action = self.actor(tensor_state).item()
        return action

    def compute_log_probs(self, states, actions):
        if states.ndim == 1:
            tensor_states = self.to_tensor(states[np.newaxis, :])
        else:
            tensor_states = self.to_tensor(states)
        tensor_actions = self.to_tensor(actions)
        log_probs = self.actor.log_prob(tensor_states, tensor_actions)
        assert log_probs.dim() == 1
        return log_probs

    def update(self, states_list, actions_list, rewards_list):
        states, rewards, actions, advantages, next_states, masks = self.process_samples(states_list, actions_list, rewards_list)
        self.update_critic(states, rewards, next_states, masks)
        self.update_actor(states, actions, advantages)

    def process_samples(self, states_list, actions_list, rewards_list):
        N = sum(len(state_list) for state_list in states_list)
        n = len(states_list[0])
        next_states = []
        masks = []
        for state_list in states_list:
            for state in state_list[1:]:
                next_states.append(state)
            next_states.append(np.zeros_like(state_list[0]))
        next_states = np.array(next_states)

        #print(next_states.shape, N, self.state_dim)
        assert next_states.shape == (N, self.state_dim)
        assert np.all(next_states[n-1] == 0)

        for state_list in states_list:
            masks += [False]*(len(state_list)-1) + [True]
        masks = np.array(masks, dtype=np.float32)
        assert masks.shape == (N,)
        assert masks[n-1] == 1

        states = np.concatenate(states_list).astype(np.float32)
        rewards = np.concatenate(rewards_list).astype(np.float32)
        actions = np.concatenate(actions_list).astype(np.float32)

        next_values = self.to_array(self.critic(self.to_tensor(next_states)).flatten())
        target = rewards + self.gamma * (1 - masks) * next_values

        values = self.to_array(self.critic(self.to_tensor(states)).flatten())

        advantages = target - values
        return states, rewards, actions, advantages, next_states, masks

    def update_critic(self, states, rewards, next_states, masks):
        tensor_states = self.to_tensor(states)
        tensor_rewards = self.to_tensor(rewards)
        tensor_next_states = self.to_tensor(next_states)
        tensor_masks = self.to_tensor(masks)
        for _ in range(self.num_critic_updates):
            self.critic.eval()
            next_values = self.critic(tensor_next_states).flatten()
            target = tensor_rewards + self.gamma * (1-tensor_masks) * next_values

            target = target.detach()
            self.critic.train()
            for _ in range(self.num_critic_update_steps):
                values = self.critic(tensor_states).flatten()
                #print(values.dtype, target.dtype)
                loss = self.critic_loss(input=values, target=target)
                self.critic_optimizer.zero_grad()
                loss.backward()

                self.critic_optimizer.step()

    def update_actor(self, states, actions, advantages):
        tensor_advantages = self.to_tensor(advantages)

        self.actor.train()
        log_probs = self.compute_log_probs(states, actions)
        assert log_probs.shape == tensor_advantages.shape

        loss = (-tensor_advantages * log_probs).mean()
        self.actor_optimizer.zero_grad()
        loss.backward()

        self.actor_optimizer.step()
        self.actor.eval()

    def save(self, path):
        torch.save(self.actor.state_dict(), path + 'a2cv0_actor.pth')
        torch.save(self.critic.state_dict(), path + 'a2cv0_critic.pth')

    def load(self, path):
        self.actor.load_state_dict(torch.load(path + 'a2cv0_actor.pth'))
        self.critic.load_state_dict(torch.load(path + 'a2cv0_critic.pth'))
