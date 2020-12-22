import gym
import gym_snake
from argparse import Namespace

from VanillaPG.pg_policy import *
from VanillaPG.pg_network import *
from VanillaPG.policy_variants import *
from utilities import *

import pandas as pd

episodes = 100
data_channels = 3 # presented pixel colors
action_space = 4
seed = 123

# Construct Environment
env = gym.make('snake-v0')
env.grid_size = [10, 10]
env.snake_size = 3
env.unit_size = 1
env.unit_gap = 0
obs = env.reset() # Constructs an instance of the game
map_width = 10
map_height = 10

net_params = {
    'lr' : 1e-3,
    'hidden_size' : 128,
    'seed' : 123,
    'gamma' : 0.9,
    'batch_size' : 10
}
# prepare network
obs_size = map_width * map_height * data_channels

policies = [PolicySM, PolicyRewards, PolicyDeep]
paths = ['./VanillaPG/128h1-sm-run2/checkpoint5000steps1092794.pkl',
        './VanillaPG/128h1-sm-negative-rewards-run2/checkpoint5000steps988924.pkl',
        './VanillaPG/128h2/checkpoint3000steps789771.pkl'] 
names = ['Default', 'NegativeStep', '2Layers']

runs = []
for polclass, path, name in zip(policies, paths, names):
    policy = polclass(obs_size, action_space, Namespace(**net_params))
    policy.load(path)
    for i in range(episodes):
        done = False
        steps = 0
        length = 0
        while not done:
            obs_norm = normalize_and_flatten(obs)
            action, _ = policy.act(obs_norm)
            obs, reward, done, info = env.step([action]) 
            steps += 1
            if reward == 1: length += 1
            if done:
                obs = env.reset()
                runs.append({
                    'name' : name,
                    'steps' : steps,
                    'snake_length': length
                })
        if i % 10 == 0:
            print(f'{i} episodes done')
    print(name + 'episodes done.')
env.close()

resdf = pd.DataFrame(runs)
print(resdf.groupby('name').agg(['mean', 'std', 'min', 'max']))


