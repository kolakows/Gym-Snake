import gym
import gym_snake
from argparse import Namespace

from pg_policy import *
from utilities import *

render_every_eps = 1000
checkpoint_every = 10000
episodes = 100000
map_width = 10
map_height = 10
data_channels = 3 # presented pixel colors
action_space = 4
seed = 123
net_params = {
    'lr' : 1e-2,
    'hidden_size' : 128,
    'seed' : 123,
    'gamma' : 0.99,
    'batch_size' : 10
}
# prepare network
obs_size = map_width * map_height * data_channels

# Construct Environment
env = gym.make('snake-v0')
env.grid_size = [10, 10]
env.snake_size = 3
env.unit_size = 1
env.unit_gap = 0

obs = env.reset() # Constructs an instance of the game

policy = PGPolicy(obs_size, action_space, Namespace(**net_params))
policy.load('./VanillaPG/snek-128h1-PG-first-take/checkpoint59999.pkl')

for _ in range(1000):
    env.render()

    obs_norm = normalize_and_flatten(obs)
    action, action_prob = policy.act(obs_norm)
    obs, reward, done, info = env.step([action]) # take a random action
    if done:
        env.reset()
env.close()


