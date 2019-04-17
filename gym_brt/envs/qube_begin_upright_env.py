from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np
from gym import spaces
from gym_brt.envs.qube_base_env import \
    QubeBaseEnv, \
    ACTION_HIGH, \
    ACTION_LOW


class QubeBeginUprightReward(object):
    def __init__(self):
        self.target_space = spaces.Box(
            low=ACTION_LOW,
            high=ACTION_HIGH, dtype=np.float32)

    def __call__(self, state, action):
        theta_x = state[0]
        theta_y = state[1]
        alpha_x = state[2]
        alpha_y = state[3]
        theta_velocity = state[4]
        alpha_velocity = state[5]

        theta = np.arctan2(theta_y, theta_x)  # arm
        alpha = np.arctan2(alpha_y, alpha_x)  # pole

        '''
        cost = theta**4 + \
            alpha**2 + \
            0.01 * alpha_velocity**2
        '''
        cost = alpha**2 + theta**2 + \
            1e-4 * alpha_velocity**2 + 1e-4 * theta_velocity**2

        reward = np.clip((15 - cost)/15.0,-0.1,1)
        return reward


class QubeBeginUprightEnv(QubeBaseEnv):
    def __init__(self, frequency=1000, use_simulator=False):
        super(QubeBeginUprightEnv, self).__init__(
            frequency=frequency,
            use_simulator=use_simulator)
        self._old_tach0 = 0
        self.reward_fn = QubeBeginUprightReward()

    def reset(self):
        # Start the pendulum stationary at the top (stable point)
        self._old_tach0 = 0
        return self.flip_up()

    def _done(self):
        # The episode ends whenever the angle alpha is outside the tolerance
        _done = np.abs(self._alpha) > (80 * np.pi / 180) or np.abs(self._tach0 - self._old_tach0) > 2000
        self._old_tach0 = self._tach0
        return _done

    def step(self, action):
        state, reward, _, info = super(QubeBeginUprightEnv, self).step(action)
        done = self._done()
        if done:
            reward = -0.5
        else:
            reward = 0.1
        return state, reward, done, info


def main():
    num_episodes = 10
    num_steps = 250

    with QubeBeginUprightEnv() as env:
        for episode in range(num_episodes):
            state = env.reset()
            for step in range(num_steps):
                action = env.action_space.sample()
                state, reward, done, _ = env.step(action)


if __name__ == '__main__':
    main()