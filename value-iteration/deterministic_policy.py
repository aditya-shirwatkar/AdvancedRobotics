import gym
from matplotlib import pyplot as plt 
import numpy as np

class Model():
    def __init__(self, type):
        super().__init__()
        self.env = gym.make('FrozenLake-v0')
        if type == 'tabular':
            self.valueFun = self.getTabularValueFun(self.env)

    class getTabularValueFun():
        def __init__(self, env):
            super().__init__()
            self.obs_dim = env.observation_space.n
            self._value_fun = np.zeros(shape=(self.obs_dim,))

        def get_values(self, states=None):
            if states is None:
                return self._value_fun
            else:
                return self._value_fun[states]

        def update(self, values):
            self._value_fun = values

    class getTabularPolicy():
        def __init__(self):
            super().__init__()
            self.action_dim = env.action_space.n
            self._policy_fun = np.zeros(shape=(self.action_dim,))

        def get_values(self, states=None):
            if states is None:
                return self._policy_fun
            else:
                return self._policy_fun[states]

        def update(self, values):
            self._value_fun = values


model = Model('tabular')

print(model.valueFun._value_fun)

print(model.env.observation_space.sample())

done = False

for i in range(1):
    while not done:
        s = model.env.reset()
        a = model.
        ns, r, done, _ = model.env.step(a)
