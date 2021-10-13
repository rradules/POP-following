import gym
import numpy as np

from gym import spaces
import scipy.sparse as sp
from gym.utils import seeding
import pickle

import random
import sys

class RandomMOMDP(gym.Env):
    def __init__(self, nstates, nobjectives, nactions, nsuccessor, seed):
        np.random.seed(seed)
        # Generate a random MOMDP, that is, a reward function and a transition function
        self._transition_function = np.zeros(shape=(nstates, nactions, nstates))
        #rew = sp.rand(nstates, nactions*nobjectives, density=density)
        #self._reward_function = rew.A.reshape(nstates, nactions, nobjectives)

        self._old_reward = np.random.rand(nstates, nactions, nobjectives)
        self._old_reward_function = np.zeros((nstates, nactions, nstates, nobjectives))
        for s1 in range(nstates):
            for a in range(nactions):
                for s2 in range(nstates):
                    for o in range(nobjectives):
                        self._old_reward_function[s1, a, s2, o] = self._old_reward[s1, a, o]
        self._reward_function = np.random.rand(nstates, nactions, nstates, nobjectives)

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

        self.info = {'states': nstates, 'objectives': nobjectives,
                'actions': nactions, 'successors': nsuccessor,
                'seed':seed, 'transition': self._transition_function.tolist(),
                'reward': self._reward_function.tolist()}

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

