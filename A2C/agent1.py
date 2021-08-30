import numpy as np
import torch
import torch.nn as nn

from common.model import Actor_Critic


class A2CTrainer_v1:
    def __init__(self, state_dim, act_dim, cfg):
        print("A2CTrainer-v1")
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.device = cfg.device
        self.clip_gradient = cfg.clip_gradient
        self.gamma = cfg.gamma
        self.hidden_dims = cfg.hidden_dims
        self.num_critic_updates = cfg.num_critic_updates
        self.num_critic_update_steps = cfg.num_critic_update_steps

        self.model = Actor_Critic(state_dim, act_dim, self.hidden_dims).to(self.device)
        self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr=cfg.lr, alpha=0.99, eps=1e-5)

    def to_tensor(self, array):
        return torch.from_numpy(array).to(self.device)

    def to_array(self, tensor):
        return tensor.cpu().detach().numpy()

    def choose_action(self, state):
        assert isinstance(state, np.ndarray)
        assert state.ndim == 1
        if state.dtype != np.float32:
            state = state.astype(np.float32)

        tensor_state = self.to_tensor(state)
        dist, _ = self.model(tensor_state)
        action = dist.sample().item()
        return action

    def compute_log_probs(self, states, actions):
        if states.ndim == 1:
            tensor_states = self.to_tensor(states[np.newaxis, :])
        else:
            tensor_states = self.to_tensor(states)
        tensor_actions = self.to_tensor(actions)
        dist, _ = self.model(tensor_states)
        log_probs = dist.log_prob(tensor_actions)
        assert log_probs.dim() == 1
        return log_probs, dist.entropy().mean()

    def update(self, states_list, actions_list, rewards_list):
        #self.model.train()
        states, rewards, actions, next_states, masks = self.process_samples(states_list, actions_list, rewards_list)

        tensor_reward = self.to_tensor(rewards)
        tensor_masks = self.to_tensor(masks)
        next_values = self.model(self.to_tensor(next_states))[1].flatten()
        target = tensor_reward + self.gamma * (1 - tensor_masks) * next_values

        values = self.model(self.to_tensor(states))[1].flatten()

        tensor_advantages = target.detach() - values
        value_loss = (tensor_advantages * tensor_advantages).mean()

        tensor_log_probs, entropy = self.compute_log_probs(states, actions)
        policy_loss = -(tensor_advantages.detach() * tensor_log_probs).mean()

        loss = value_loss + 0.5 * policy_loss - 0.001 * entropy
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        #self.model.eval()

    def process_samples(self, states_list, actions_list, rewards_list):
        N = sum(len(state_list) for state_list in states_list)
        n = len(states_list[0])
        next_states = []
        masks = []
        for state_list in states_list:
            for state in state_list[1:]:
                next_states.append(state)
            next_states.append(np.zeros_like(state_list[0]))
        next_states = np.array(next_states, dtype=np.float32)

        assert next_states.shape == (N, self.state_dim)
        assert np.all(next_states[n - 1] == 0)

        for state_list in states_list:
            masks += [False] * (len(state_list) - 1) + [True]
        masks = np.array(masks, dtype=np.float32)
        assert masks.shape == (N,)
        assert masks[n - 1] == 1

        states = np.concatenate(states_list).astype(np.float32)
        rewards = np.concatenate(rewards_list).astype(np.float32)
        actions = np.concatenate(actions_list).astype(np.float32)

        return states, rewards, actions, next_states, masks

    def save(self, path):
        torch.save(self.model.state_dict(), path + 'a2cv1.pth')

    def load(self, path):
        self.model.load_state_dict(torch.load(path + 'a2cv1.pth'))

