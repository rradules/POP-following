import random
import numpy as np
from collections import deque


class ReplayBuffer:
    """
    A class implementing a simple replay buffer.
    """
    def __init__(self, capacity=50000, batch_size=64):
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity
        self.batch_size = batch_size

    def append(self, new_vec, N, state, action, next_state):
        """
        Add an experience to the buffer.
        :param new_vec: The new vector to follow from this experience.
        :param N: The N vector from the experience.
        :param state: The state from the experience.
        :param action: The action from the experience.
        :param next_state: The next state from the experience.
        :return: /
        """
        entry = []  # We first flatten all inputs so that we can later easily extract them.
        entry.extend(new_vec)
        entry.extend(N)
        entry.extend([state, action, next_state])
        self.buffer.append(entry)

    def sample(self):
        """
        Sample experiences from the from the buffer.
        :return: A batch of experiences.
        """
        return np.array(random.sample(self.buffer, self.batch_size))

    def can_sample(self):
        return self.size() >= self.batch_size

    def size(self):
        """
        This method gets the length of the buffer.
        :return: The size of the buffer.
        """
        return len(self.buffer)
