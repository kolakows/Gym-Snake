import gym
import gym_snake
from argparse import Namespace

from VanillaPG.pg_policy import *
from VanillaPG.pg_network import *
from utilities import *

episodes = 100000
render_every_eps = 1000
checkpoint_every = 10000
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
    'lr' : 1e-2,
    'hidden_size' : 128,
    'seed' : 123,
    'gamma' : 0.99,
    'batch_size' : 10
}
# prepare network
PGNetwork = PGNetwork_sm
obs_size = map_width * map_height * data_channels
policy = PGPolicy(obs_size, action_space, PGNetwork, Namespace(**net_params))
policy.load('./VanillaPG/checkpoints/checkpoint5000.pkl')

for i in range(1000):
    obs_norm = normalize_and_flatten(obs)
    action, _ = policy.act(obs_norm)
    if i > 0: policy.render_probs(env.fig)
    env.render(frame_speed=1.)
    obs, reward, done, info = env.step([action]) 
    if done:
        env.reset()
env.close()


