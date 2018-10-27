"""
UnityML Environment.
"""

import platform
import numpy as np
from unityagents import UnityEnvironment

class UnityMLVectorMultiAgent():
    """Multi-agent UnityML environment with vector observations."""

    def __init__(self, evaluation_only=False, seed=0):
        """Load platform specific file and initialize the environment."""
        os = platform.system()
        if os == 'Darwin':
            file_name = 'Soccer.app'
        elif os == 'Linux':
            file_name = 'Soccer_Linux/Soccer.x86_64'
        self.env = UnityEnvironment(file_name='unity_envs/' + file_name, seed=seed)
        self.brain_names = self.env.brain_names
        self.evaluation_only = evaluation_only

    def reset(self):
        """Reset the environment."""
        info = self.env.reset(train_mode=not self.evaluation_only)
        state0 = info[self.brain_names[0]].vector_observations
        state1 = info[self.brain_names[1]].vector_observations
        state = np.vstack((state0, state1))
        return state

    def step(self, action):
        """Take a step in the environment."""
        actions = action.reshape(2, 2)
        action = {'GoalieBrain': actions[0], 'StrikerBrain': actions[1]}
        info = self.env.step(action)

        state0 = info[self.brain_names[0]].vector_observations
        state1 = info[self.brain_names[1]].vector_observations
        state = np.vstack((state0, state1))
        reward0 = info[self.brain_names[0]].rewards
        reward1 = info[self.brain_names[1]].rewards
        reward = reward0 + reward1
        done0 = info[self.brain_names[0]].local_done
        done1 = info[self.brain_names[1]].local_done
        done = done0 + done1
        return state, reward, done
