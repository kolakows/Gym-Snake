import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as T
import numpy as np

import gym
import gym_snake

class PGNetwork(nn.Module):
    def __init__(self, obs_space_size, hidden_size, action_space_size, seed):       
        super().__init__()
        torch.manual_seed(seed)
        self.h1 = nn.Linear(obs_space_size, hidden_size)
        self.out = nn.Linear(hidden_size, action_space_size)

    def forward(self, x):
        x = F.relu(self.h1(x))
        x = F.log_softmax(self.out(x), dim = -1)
        return x

class PGNetwork_sm(nn.Module):
    def __init__(self, obs_space_size, hidden_size, action_space_size, seed):       
        super().__init__()
        torch.manual_seed(seed)
        self.h1 = nn.Linear(obs_space_size, hidden_size)
        self.out = nn.Linear(hidden_size, action_space_size)

    def forward(self, x):
        x = F.relu(self.h1(x))
        x = F.softmax(self.out(x), dim = -1)
        return x


class PGNetwork_deep(nn.Module):
    def __init__(self, obs_space_size, hidden_size, hidden_count, action_space_size, seed):       
        super().__init__()
        torch.manual_seed(seed)
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(obs_space_size, hidden_size))
        for _ in range(hidden_count-1):
            self.layers.append(nn.Linear(hidden_size, hidden_size))
        self.layers.append(nn.Linear(hidden_size, action_space_size))

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        x = F.softmax(self.layers[-1](x), dim = -1)
        return x
