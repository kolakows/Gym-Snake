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
        x = F.softmax(self.out(x), dim = -1)
        return x

class PGNetwork_deep(nn.Module):
    def __init__(self, obs_space_size, hidden_size, action_space_size, seed, hidden_count = 2):       
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


class PGSimpleConvNet(nn.Module):
    def __init__(self, input_size, action_space_size, seed):
        '''
        Weights = 3 * 3 * 3 * 3 * 4 + 12 * 32 * 4 = 324 + 1536 = 1860 | channels (3) * filter size (3x3x3) * num of layers (4) + FC last layer
        
        actually it is a filter 3x3x3 :o

        add zero padding, for corners to be treaded equally? 
        '''
        super().__init__()
        torch.manual_seed(seed)
        self.conv1 = nn.Conv2d(3, 3, 3) # 3x8x8 out
        self.conv2 = nn.Conv2d(3, 3, 3) # 3x6x6 out
        self.conv3 = nn.Conv2d(3, 3, 3) # 3x4x4 out
        self.conv4 = nn.Conv2d(3, 3, 3) # 3x2x2 out
        self.h1 = nn.Linear(12, 32)
        self.out = nn.Linear(32, 4)
    def forward(self, x):
        x = x.unsqueeze(0)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.h1(x.flatten()))
        x = F.softmax(self.out(x))
        return x


# class PGConvNet(nn.Module):
#     def __init__(self, input_size, action_space_size, seed):
#         '''
#         Weights = 3 * 9 * 4 + 12 * 32 * 4 = 108 + 1536 = 1648 | channels * filter size * num of layers + FC last layer
#         '''
#         super().__init__()
#         torch.manual_seed(seed)
#         self.conv1 = nn.Conv2d(3, 3, 3) # 3x8x8 out
#         # first print some gradient values but before update overleaf doc
#         self.conv2 = nn.Conv2d(3, 3, 3) # 3x6x6 out
#         self.conv3 = nn.Conv2d(3, 3, 3) # 3x4x4 out
#         self.conv4 = nn.Conv2d(3, 3, 3) # 3x2x2 out
#         self.h1 = nn.Linear(12, 32)
#         self.out = nn.Linear(32, 4)
#     def forward(self, x):
#         x = x.unsqueeze(0)
#         x = F.relu(self.conv1(x))
#         x = F.relu(self.conv2(x))
#         x = F.relu(self.conv3(x))
#         x = F.relu(self.conv4(x))
#         x = F.relu(self.h1(x.flatten()))
#         x = F.softmax(self.out(x))
#         return x