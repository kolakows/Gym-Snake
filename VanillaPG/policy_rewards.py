
from VanillaPG.pg_policy import PGPolicy
from VanillaPG.pg_network import PGNetwork

class PolicyRewards(PGPolicy):
    def __init__(self, obs_size, action_size, parameters):
        super().__init__(obs_size, action_size, PGNetwork, parameters)
    
    def step(self, action_log_prob, reward, done):
        penalty = -0.01
        super().step(action_log_prob, reward + penalty, done)