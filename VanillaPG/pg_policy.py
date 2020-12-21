import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as T
from torch.distributions import Categorical

import numpy as np
import gym
import gym_snake


# check only running updates on episodes with high loss?
# check input as difference between states

class PGPolicy:
    '''
    good intro on PG - pong from pixels
    https://www.youtube.com/watch?v=tqrcjHuNdmQ
    code gist
    https://gist.github.com/karpathy/a4166c7fe253700972fcbc77e4ea32c5
    '''
    def __init__(self, obs_size, action_size, PGNetwork, parameters):
        self.pgnetwork = PGNetwork(obs_size, parameters.hidden_size, action_size, parameters.seed)
        self.rng = np.random.default_rng(parameters.seed)
        self.action_size = action_size
        self.lr = parameters.lr
        self.gamma = parameters.gamma
        self.batch_size = parameters.batch_size
        self.buffered_episodes = 0
        self.eps = np.finfo(np.float32).eps.item()
        # for tracking action probs text on env fig
        self.probtxt = None

        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # # Device
        # if parameters.use_gpu and torch.cuda.is_available():
        #     self.device = torch.device("cuda:0")
        #     print("🐇 Using GPU")
        # else:
        #     self.device = torch.device("cpu")
        #     print("🐢 Using CPU")

        self.optimizer = optim.Adam(self.pgnetwork.parameters(), lr = self.lr)

        # save info about taken action probability and received reward during episode
        self.saved_log_probs = []
        self.rewards = []

    def step(self, action_log_prob, reward, done):
        '''
        Store experiences, learn every batch_size epizodes, when the episode buffer is ready
        '''
        # operate on diffs? what about inference from state?
        # state_diff = state-prev_state
        if not done:
            self.saved_log_probs.append(action_log_prob)
            self.rewards.append(reward)
        else:
            log_probs = torch.stack(self.saved_log_probs)    
            returns = []
            # calculate discounted rewards
            R = 0
            for r in self.rewards[::-1]:
                R = r + self.gamma * R
                returns.insert(0, R)
            returns = torch.tensor(returns)
            
            # center and scale returns
            returns = (returns - returns.mean()) / (returns.std() + self.eps)

            # modulate loss with advantage ('PG magic happens here')
            loss = - log_probs * returns
            loss = loss.sum()

            loss.backward()
            self.buffered_episodes += 1
            self.saved_log_probs = []
            self.rewards = []
      
        # perform network parameters' update
        if self.buffered_episodes == self.batch_size:
            print(f'Updated weights ')
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.buffered_episodes = 0
           
            
    def act(self, obs):
        '''
        Performs inference on internal neural network, which returns action probabilities, then draws action
        returns chosen action, log probability of choosing that action
        '''
        # numpy to tensor
        obs_norm_torch = torch.from_numpy(obs)
        obs_norm_torch.requires_grad = True

        self.action_probs = self.pgnetwork(obs_norm_torch)
        # Categorical and then .log_prob allow for backpropagation through computation graph, numpy obviously doesnt track that
        # action = self.rng.choice(np.arange(self.action_size), p = action_probs)
        m = Categorical(self.action_probs)
        action = m.sample()
        return action, m.log_prob(action) 

    def save(self, filename):
        torch.save({'model_state_dict': self.pgnetwork.state_dict()}, filename)

    def load(self, filename):
        checkpoint = torch.load(filename)
        self.pgnetwork.load_state_dict(checkpoint['model_state_dict'])

    def render_probs(self, envfig):
        adict = {
            'UP' : 0,
            'RIGHT' : 1,
            'DOWN' : 2,
            'LEFT' : 3,
            }
        txt = []
        for k, v in adict.items():
            txt.append(f'{k:5s} : {self.action_probs[v]:.2f}\n')
        aprobtext = ''.join(txt)
        if self.probtxt:
            self.probtxt.remove()
        self.probtxt = envfig.text(0.02, 0.5, aprobtext, fontsize = 12)

