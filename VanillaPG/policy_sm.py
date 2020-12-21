
from VanillaPG.pg_policy import PGPolicy
from VanillaPG.pg_network import PGNetwork

class PolicySM(PGPolicy):
    def __init__(self, obs_size, action_size, parameters):
        super().__init__(obs_size, action_size, PGNetwork, parameters)
