"""some network for actor or critic"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class MLP(nn.Module):
    """multiple layers perceptron"""
    def __init__(self, input_dim, output_dim, hidden_dim=128):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = x.type_as(self.fc1.bias)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class MLP_Categorical(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=36):
        super(MLP_Categorical, self).__init__()
        self.net = MLP(input_dim, output_dim, hidden_dims)

    def forward(self, x):
        logits = self.net(x)
        dist = torch.distributions.Categorical(logits=logits)
        return dist.sample()

    def log_prob(self, obs, act):
        logits = self.net(obs)
        dist = torch.distributions.Categorical(logits=logits)
        return dist.log_prob(act)


class DuelingModel(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=128):
        super(DuelingModel, self).__init__()
        self.feature = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )
        self.advantage = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        self.value = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        x = self.feature(x)
        advantage = self.advantage(x)
        value = self.value(x)
        assert advantage.ndim == value.ndim
        return value + advantage - advantage.mean()


class Actor_Critic(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=36):
        super(Actor_Critic, self).__init__()

        self.critic = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        self.actor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        value = self.critic(x)
        logits = self.actor(x)
        dist = Categorical(logits=logits)
        return dist, value


class DDPGActor(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, init_w=3e-3):
        super(DDPGActor, self).__init__()
        self.linear0 = nn.Linear(input_dim, hidden_dim)
        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, output_dim)

        self.linear2.weight.data.uniform_(-init_w, init_w)
        self.linear2.bias.data.uniform_(-init_w, init_w)

    def forward(self, x):
        x = F.relu(self.linear0(x))
        x = F.relu(self.linear1(x))
        x = torch.tanh(self.linear2(x))
        #映到(-1,1)
        return x


class DDPGCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim, init_w=3e-3):
        super(DDPGCritic, self).__init__()
        self.linear0 = nn.Linear(obs_dim+act_dim, hidden_dim)
        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, 1)

        self.linear2.weight.data.uniform_(-init_w, init_w)
        self.linear2.weight.data.uniform_(-init_w, init_w)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.linear0(x))
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x