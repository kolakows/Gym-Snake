import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as T
import numpy as np
import gym
import gym_snake
import os
from argparse import Namespace

from pg_network import *
from VanillaPG.pg_policy import *
from VanillaPG.policy_variants import *
from utilities import *

# choose policy to train

policyConstructor = PolicySM
#policyConstructor = PolicyRewards
#policyConstructor = PolicyDeep

render = False
render_every_eps = 1000
checkpoint_every = 1000
episodes = 100000
map_width = 10
map_height = 10
data_channels = 3 # presented pixel colors
action_space = 4
seed = 123

# prepare network
net_params = {
    'lr' : 1e-3,
    'hidden_size' : 128,
    'seed' : 123,
    'gamma' : 0.9,
    'batch_size' : 10
}
obs_size = map_width * map_height * data_channels
policy = policyConstructor(obs_size, action_space, Namespace(**net_params))

# Construct Environment
env = gym.make('snake-v0')
env.grid_size = [map_width, map_height]
env.snake_size = 3
# make obs [height, width, channels]
env.unit_size = 1
env.unit_gap = 0

snake_len = env.snake_size
obs = env.reset() # Constructs an instance of the game
total_steps = 0
for episode in range(episodes):
    done = False
    steps = 0
    while not done:
        if render: env.render()    
        obs_norm = normalize_and_flatten(obs)
        action, action_prob = policy.act(obs_norm)
        # progress env
        obs, reward, done, info = env.step([action])
        # update memory buffer and learn
        policy.step(action_prob, reward, done)
        if reward == 1:
            snake_len += 1
        steps += 1
        total_steps += 1
    print(f'Episode {episode} finished, {steps} env steps. Final snake length {snake_len}')
    env.reset()
    snake_len = env.snake_size
    if episode % render_every_eps == 0:
        render = True
    else:
        render = False
    # store model
    if episode % checkpoint_every == checkpoint_every-1:
        model_path = './VanillaPG/checkpoints/'
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        policy.save(model_path + 'checkpoint'+ str(episode+1)+'steps' + str(total_steps) + '.pkl')
        
env.close()





