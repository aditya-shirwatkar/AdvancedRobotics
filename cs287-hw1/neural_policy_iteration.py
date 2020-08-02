import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gym
import mujoco_py
import gym_custom_envs
import numpy as np
from matplotlib import pyplot as plt
import random
from collections import namedtuple

config = {'function approximator': 'neural network',
          'environment': 'cartpole'}

# initialise reward graphs
xval = []
yval = []

# make a named tuple for sample states
Experience = namedtuple('Experience', ('state', 'action', 'reward', 'next_state', 'done'))

if config['environment'] == 'cartpole':
    env = gym.make('InvertedPendulum-v2')

print('observations range ==', env.observation_space.low, env.observation_space.high)
print('actions range ==', env.action_space.low, env.action_space.high)

if config['function approximator'] == 'neural network':
    # Make a neural network architecture, the below one gave decent results
    class FunctionApproximator(nn.Module):
        def __init__(self):
            super().__init__()
            self.input = (env.observation_space.low).shape(0)
            self.fc1 = nn.Linear(4, 16)
            self.fc2 = nn.Linear(16, 8)
            self.fc3 = nn.Linear(8, 2)

        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = (self.fc3(x))
            return x
