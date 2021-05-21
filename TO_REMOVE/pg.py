import os
import sys
import argparse

import keras
import keras.backend as K
import gym
import numpy as np
import datetime

from gym.envs.registration import register

epsilon = 1e-8

register(
    id='RandomMOMDP-v0',
    entry_point='randommomdp:RandomMOMDP',
    reward_threshold=0.0,
    kwargs={'nstates': 100, 'nobjectives': 4, 'nactions': 8, 'nsuccessor': 12, 'seed': 1}
)
register(
    id='FishWood-v0',
    entry_point='fishwood:FishWood',
    reward_threshold=0.0,
    kwargs={'fishproba': 0.1, 'woodproba': 0.9}
)

class Experience(object):
    def __init__(self, state, action):
        """ In <state>, <action> has been choosen, which led to <reward>
        """
        self.state = state
        self.action = action
        self.rewards = None             # The rewards are set once they are known
        self.interrupt = False          # Set to true at the end of an episode

class Learner(object):
    def __init__(self, args):
        """ Construct a Learner from parsed arguments
        """

        # Make environment
        self._env = gym.make(args.env)
        self._render = args.render
        self._return_type = args.ret
        self._extra_state = args.extra_state

        # Native actions
        aspace = self._env.action_space

        if isinstance(aspace, gym.spaces.Tuple):
            aspace = aspace.spaces
        else:
            aspace = [aspace]               # Ensure that the action space is a list for all the environments

        self._num_rewards = getattr(self._env, 'num_rewards', 1)
        self._num_actions = np.prod([a.n for a in aspace])
        self._aspace = aspace

        # Make an utility function
        if args.utility is not None:
            self._utility = compile(args.utility, 'utility', 'eval')
        else:
            self._utility = None

        # Build network
        self._discrete_obs = isinstance(self._env.observation_space, gym.spaces.Discrete)

        if self._discrete_obs:
            self._state_vars = self._env.observation_space.n                    # Prepare for one-hot encoding
        else:
            self._state_vars = np.product(self._env.observation_space.shape)

        if self._extra_state == 'none':
            self._actual_state_vars = self._state_vars
        elif self._extra_state == 'timestep':
            self._actual_state_vars = self._state_vars + 1                      # Add the timestep to the state
        elif self._extra_state == 'accrued':
            self._actual_state_vars = self._state_vars + self._num_rewards      # Accrued vector reward
        elif self._extra_state == 'both':
            self._actual_state_vars = self._state_vars + self._num_rewards + 1  # Both addition

        self.make_network(args.hidden, args.lr)

        print('Number of primitive actions:', self._num_actions)
        print('Number of state variables', self._actual_state_vars)
        print('Number of objectives', self._num_rewards)

        # Lists for policy gradient
        self._experiences = []

    def make_network(self, hidden, lr):
        """ Initialize a simple multi-layer perceptron for policy gradient
        """
        # Useful functions
        def make_probas(inputs):
            pi = inputs

            # Normalized sigmoid. Gives better results than Softmax
            x_exp = K.sigmoid(pi)
            return x_exp / K.sum(x_exp)

        def make_function(input, noutput, activation='sigmoid'):
            dense1 = keras.layers.Dense(units=hidden, activation='tanh')(input)
            dense2 = keras.layers.Dense(units=noutput, activation=activation)(dense1)

            return dense2

        # Neural network with state as input and a probability distribution over
        # actions as output
        state = keras.layers.Input(shape=(self._actual_state_vars,))

        pi = make_function(state, self._num_actions, 'linear')                  # Option to execute given current state and option
        probas = keras.layers.core.Lambda(make_probas, output_shape=(self._num_actions,))(pi)

        self._model = keras.models.Model(inputs=[state], outputs=[probas])

        # Compile model with Policy Gradient loss
        print("Compiling model", end="")
        self._model.compile(optimizer=keras.optimizers.Adam(lr=lr), loss='mse')
        print(" done")

        # Policy gradient loss for the policy
        pi_true = self._model.targets[0]
        pi_pred = self._model.outputs[0]
        logpi = K.log(pi_pred + epsilon)
        grad = K.mean(pi_true * logpi)

        self._model.total_loss = -grad

    def encode_state(self, state, timestep, accrued):
        """ Encode a raw state from Gym to a Numpy vector
        """
        if self._discrete_obs:
            # One-hot encode discrete variables
            rs = np.zeros(shape=(self._state_vars,))
            rs[state] = 1.0
        elif isinstance(state, np.ndarray):
            rs = state.flatten()
        else:
            rs = np.array(state)

        # Add the extra state information
        extratimestep = [(50 - timestep) * 0.1]
        extraaccrued = accrued * 0.1

        if self._extra_state == 'timestep':
            return np.append(rs, extratimestep)
        elif self._extra_state == 'accrued':
            return np.append(rs, extraaccrued)
        elif self._extra_state == 'both':
            return np.append(rs, np.append(extratimestep, extraaccrued))
        else:
            return rs
    
    def encode_reward(self, reward):
        """ Encode a scalar or vector reward as an array
        """
        if ifinstance(reward, float):
            return np.array([reward])
        else:
            return np.array(reward)

    def predict_probas(self, state):
        """ Return a probability distribution over actions given the current state.
        """
        output = self._model.predict_on_batch([
            state.reshape((1, self._actual_state_vars))
        ])

        return output[0].flatten()

    def scalarize_reward(self, rewards):
        """ Return a scalarized reward from objective scores
        """
        if self._utility is None:
            # Default scalarization, just a sum
            return np.sum(rewards)
        else:
            # Use the user utility function
            return eval(self._utility, {}, {'r'+str(i+1): rewards[i] for i in range(self._num_rewards)})

    def learn_from_experiences(self):
        """ Learn from the experience pool, using Policy Gradient
        """
        N = len(self._experiences)

        if N == 0:
            return

        target_action = np.zeros(shape=(N, self._num_actions))
        source_state = np.zeros(shape=(N, self._actual_state_vars))

        # Compute forward-looking cumulative rewards
        forward_cumulative_rewards = np.zeros(shape=(N, self._num_rewards))
        backward_cumulative_rewards = np.zeros(shape=(N, self._num_rewards))
        cumulative_reward = np.zeros(shape=(1, self._num_rewards))

        for i in range(N-1, -1, -1):
            e = self._experiences[i]

            if e.interrupt:
                cumulative_reward.fill(0.0)     # Break the cumulative reward chain

            cumulative_reward += e.rewards
            forward_cumulative_rewards[i] = cumulative_reward

        # Compute the backward-looking cumulative reward
        cumulative_reward.fill(0.0)

        for i in range(N):
            e = self._experiences[i]

            cumulative_reward += e.rewards
            backward_cumulative_rewards[i] = cumulative_reward

            if e.interrupt:
                cumulative_reward.fill(0.0)

        # Build source and target arrays for the actor
        for i in range(N):
            e = self._experiences[i]

            # Scalarize the return
            value = self.scalarize_reward(forward_cumulative_rewards[i])

            if self._return_type == 'both':
                value += self.scalarize_reward(backward_cumulative_rewards[i])

            target_action[i, e.action] = value
            source_state[i, :] = e.state

        # Train the neural network
        self._model.fit(
            [source_state],
            [target_action],
            batch_size=N,
            epochs=1,
            verbose=0
        )

        # Prepare for next episode
        self._experiences.clear()

    def run(self):
        """ Execute an option on the environment
        """
        env_state = self._env.reset()

        done = False
        cumulative_rewards = np.zeros(shape=(self._num_rewards,))
        timestep = 0

        while not done:
            timestep += 1

            # Select an action or option based on the current state
            old_env_state = env_state
            state = self.encode_state(env_state, timestep, cumulative_rewards)

            probas = self.predict_probas(state)
            action = np.random.choice(self._num_actions, p=probas)

            # Store experience, without the reward, that is not yet known
            e = Experience(
                state,
                action
            )
            self._experiences.append(e)

            # Execute the action
            if len(self._aspace) > 1:
                # Choose each of the factored action depending on the composite action
                actions = [0] * len(self._aspace)
                a = action

                for i in range(len(actions)):
                    actions[i] = a % self._aspace[i].n
                    a //= self._aspace[i].n

                env_state, rewards, done, __ = self._env.step(actions)
            else:
                # Simple scalar action
                env_state, rewards, done, __ = self._env.step(action)

            if self._render:
                self._env.render()

            # Update the experience with its reward
            cumulative_rewards += rewards
            e.rewards = rewards

        # Mark episode boundaries
        self._experiences[-1].interrupt = True

        return cumulative_rewards

def main():
    # Parse parameters
    parser = argparse.ArgumentParser(description="Reinforcement Learning for the Gym")

    parser.add_argument("--render", action="store_true", default=False, help="Enable a graphical rendering of the environment")
    parser.add_argument("--monitor", action="store_true", default=False, help="Enable Gym monitoring for this run")
    parser.add_argument("--env", required=True, type=str, help="Gym environment to use")
    parser.add_argument("--avg", type=int, default=1, help="Episodes run between gradient updates")
    parser.add_argument("--episodes", type=int, default=1000, help="Number of episodes to run")
    parser.add_argument("--name", type=str, default='', help="Experiment name")

    parser.add_argument("--ret", type=str, choices=['forward', 'both'], default='both', help='Type of return used for training, only forward-looking or also using accumulated rewards')
    parser.add_argument("--utility", type=str, help="Utility function, a function of r1 to rN")
    parser.add_argument("--extra-state", type=str, choices=['none', 'timestep', 'accrued', 'both'], default='none', help='Additional information given to the agent, like the accrued reward')
    parser.add_argument("--hidden", default=50, type=int, help="Hidden neurons of the policy network")
    parser.add_argument("--lr", default=1e-3, type=float, help="Learning rate of the neural network")

    # Next and Sub from arguments
    args = parser.parse_args()

    # Instantiate learner
    learner = Learner(args)

    # Learn
    f = open('out-' + args.name, 'w')

    if args.monitor:
        learner._env.monitor.start('/tmp/monitor', force=True)

    try:
        old_dt = datetime.datetime.now()
        avg = np.zeros(shape=(learner._num_rewards,))

        for i in range(args.episodes):
            rewards = learner.run()

            if i == 0:
                avg = rewards
            else:
                avg = 0.99 * avg + 0.01 * rewards

            # Learn when enough experience is accumulated
            if (i % args.avg) == 0:
                learner.learn_from_experiences()

            scalarized_avg = learner.scalarize_reward(avg)

            print("Cumulative reward:", rewards, "; average rewards:", avg, scalarized_avg, file=f)
            print(args.name, "Cumulative reward:", rewards, "; average rewards:", avg, scalarized_avg)
            f.flush()
    except KeyboardInterrupt:
        pass

    if args.monitor:
        learner._env.monitor.close()

    f.close()

if __name__ == '__main__':
    main()
