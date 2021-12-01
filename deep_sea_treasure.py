from gym.envs.toy_text import discrete
import numpy as np
import cv2

UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3

UP_MOVE = [-1, 0]
DOWN_MOVE = [1, 0]
RIGHT_MOVE = [0, 1]
LEFT_MOVE = [0, -1]
ACTIONS = [UP_MOVE, DOWN_MOVE, RIGHT_MOVE, LEFT_MOVE]


class DeepSeaTreasureEnv(discrete.DiscreteEnv):
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, width=10, noise=0.0, seed=1):
        np.random.seed(seed=seed)
        self.shape = (width + 1, width)
        self.start_state_index = 0

        nS = int(np.prod(self.shape))
        nA = 4
        nO = 2

        # Calculate transition probabilities and rewards
        treasures = self._treasures()
        P = {}  # Transitions and rewards as required by gym
        self._transition_function = np.zeros((nS, nA, nS))  # Cleaner code for transitions.
        self._reward_function = np.zeros((nS, nA, nS, nO))  # And cleaner code for rewards.
        for s in range(nS):
            position = np.unravel_index(s, self.shape)
            P[s] = {a: [] for a in range(nA)}
            P[s][UP] = self._calculate_transition_prob(position, UP_MOVE, noise)
            P[s][RIGHT] = self._calculate_transition_prob(position, RIGHT_MOVE, noise)
            P[s][DOWN] = self._calculate_transition_prob(position, DOWN_MOVE, noise)
            P[s][LEFT] = self._calculate_transition_prob(position, LEFT_MOVE, noise)

            # Build up the transition and reward functions.
            if s not in treasures:  # If the current state is a treasure, make it terminal by leaving everything zero.
                for action in range(nA):
                    transitions = P[s][action]
                    for transition in transitions:
                        prob = transition[0]
                        next_state = transition[1]
                        reward = transition[2]
                        self._transition_function[s, action, next_state] = prob
                        self._reward_function[s, action, next_state] = reward

        # Calculate initial state distribution
        # We always start in state (0, 0)
        isd = np.zeros(nS)
        isd[self.start_state_index] = 1.0

        super(DeepSeaTreasureEnv, self).__init__(nS, nA, P, isd)

    def _treasures(self):

        if self.shape[1] > 10:
            raise ValueError('Default Deep Sea Treasure only supports a grid-size of max 10')

        return {(1, 0): 1,
                (2, 1): 2,
                (3, 2): 3,
                (4, 3): 5,
                (4, 4): 8,
                (4, 5): 16,
                (7, 6): 24,
                (7, 7): 50,
                (9, 8): 74,
                (10, 9): 124}

    def _unreachable_positions(self):
        u = []
        treasures = self._treasures()
        for p in treasures.keys():
            for i in range(p[0] + 1, self.shape[0]):
                u.append((i, p[1]))
        return u

    def _limit_coordinates(self, coord):
        """
        Prevent the agent from falling out of the grid world
        :param coord:
        :return:
        """
        coord[0] = min(coord[0], self.shape[0] - 1)
        coord[0] = max(coord[0], 0)
        coord[1] = min(coord[1], self.shape[1] - 1)
        coord[1] = max(coord[1], 0)
        return coord

    def _calculate_transition_prob(self, current, delta, noise):
        unreachable = self._unreachable_positions()
        treasures = self._treasures()
        intended_position = np.array(current) + np.array(delta)
        intended_position = self._limit_coordinates(intended_position).astype(int)
        intended_position = tuple(intended_position)
        if intended_position in unreachable:
            intended_position = tuple(current)
        intended_position = np.ravel_multi_index(intended_position, self.shape)

        reachable = {}
        for action in ACTIONS:
            new_position = np.array(current) + np.array(action)
            new_position = self._limit_coordinates(new_position).astype(int)
            new_position = tuple(new_position)

            if new_position in unreachable:
                new_position = tuple(current)
            new_state = np.ravel_multi_index(new_position, self.shape)

            if new_position in treasures:
                reward = [treasures[new_position], -1]
                done = True
            else:
                reward = [0, -1]
                done = False
            reachable[new_state] = (np.array(reward), done)

        transitions = []
        if len(reachable) == 1:
            for state, returns in reachable.items():
                transitions.append((1., state, returns[0], returns[1]))
        else:
            noise_prob = noise / (len(reachable) - 1)
            for state, returns in reachable.items():
                if state == intended_position:
                    transitions.append((1-noise, state, returns[0], returns[1]))
                else:
                    transitions.append((noise_prob, state, returns[0], returns[1]))
        return transitions

    def render(self, mode='rgb_array'):
        tile_size = 30
        img = np.full((self.shape[0] * tile_size, self.shape[1] * tile_size, 3), 255, np.uint8)

        y = np.tile(np.arange(tile_size, (self.shape[0] + 1) * tile_size, tile_size), self.shape[1])
        x = np.repeat(np.arange(tile_size, (self.shape[1] + 1) * tile_size, tile_size), self.shape[0])
        for x_i in x:
            for y_i in y:
                cv2.circle(img, (x_i, y_i), 0, (255, 0, 0))

        for c, t in self._treasures().items():
            cv2.putText(img, str(t), (tile_size * c[1] + tile_size // 2, tile_size * c[0] + tile_size // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, .2, 255)
        position = np.unravel_index(self.s, self.shape)
        cv2.putText(img, 'sub', (tile_size * position[1] + tile_size // 2, tile_size * position[0] + tile_size // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, .2, 255)

        return img


class BountyfulSeaTreasureEnv(DeepSeaTreasureEnv):

    def __init__(self, width=10):
        # random treasure-depths for each x-pos
        depths = np.random.choice(range(4), size=width - 1, p=[.3, .5, .1, .1])
        # add first treasure depth (always 1)
        depths = np.append([1], depths)
        depths = np.cumsum(depths)
        # limit to grid
        depths[depths > width] = width
        self.depths = depths
        super(BountyfulSeaTreasureEnv, self).__init__(width=width)

    def _treasures(self):
        pareto_front = lambda x: np.round(-45.64496 - (59.99308 / -0.2756738) * (1 - np.exp(0.2756738 * x)))

        return {(d, i): pareto_front(-(i + d)) for i, d in enumerate(self.depths)}
