#%%
import torch
import torch.nn as nn
from agent import GNN
#%%

class PPO:
    def __init__(self, env):
        # normally extract env info, but env is unlimited here
        self.actor = GNN()
        self.critic = GNN()
        # TODO: extract node info though
    def learn(self):
        kk