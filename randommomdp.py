import gym
import numpy as np

from gym import spaces
import scipy.sparse as sp
from gym.utils import seeding

import random
import sys

class RandomMOMDP(gym.Env):
    def __init__(self, nstates, nobjectives, nactions, nsuccessor, seed):
        np.random.seed(seed)

        # Generate a random MOMDP, that is, a reward function and a transition function
        self._transition_function = np.zeros(shape=(nstates, nactions, nstates))
        rew = sp.rand(nstates, nactions*nobjectives, density=0.05)
        self._reward_function = rew.A.reshape(nstates, nactions, nobjectives)
        print(self._reward_function)
        #self._reward_function = np.random.rand(nstates, nactions, nobjectives)

        # Ensure that every state has at most nsuccessor successors
        for s in range(nstates):
            successor_indexes = np.random.random_integers(0, nstates-1, nsuccessor)

            for a in range(nactions):
                self._transition_function[s, a, successor_indexes] = np.random.rand(nsuccessor)

        # Normalize the transition function over the last dimension: s, a => norm(s)
        self._transition_function /= np.sum(self._transition_function, axis=2)[:,:,None]

        self.action_space = spaces.Discrete(nactions)
        self.observation_space = spaces.Discrete(nstates)
        self.num_rewards = nobjectives

        self._nstates = nstates
        self._nobjectives = nobjectives
        self._nactions = nactions

        self.seed()
        self.reset()

    def reset(self):
        """ Reset the environment and return the initial state number
        """
        # Pick an initial state at random
        self._state = random.randrange(self._nstates)
        self._timestep = 0

        return self._state

    def step(self, action):
        # Change state using the transition function
        rewards = self._reward_function[self._state, action]

        self._state = np.random.choice(self._nstates, p=self._transition_function[self._state, action])
        self._timestep += 1

        # Return the current state, a reward and whether the episode terminates
        return self._state, rewards, self._timestep == 50, {}
