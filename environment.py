"""
UnityML Environment.

1st is red
2nd is blue

GoalieBrain:
0: forward
1: backward
2: slide right
3: slide left

StrikerBrain:
0: forward
1: backward
2: spin right (clockwise)
3: spin left (counter-clockwise)
4: slide left
5: slide right

Accepts range of values, e.g. between [0 and 1) for forward.
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
        #info = self.env.reset(train_mode=self.evaluation_only)
        state0 = info[self.brain_names[0]].vector_observations
        state1 = info[self.brain_names[1]].vector_observations
        state = np.vstack((state0, state1))
        return state

    def step(self, action):
        """Take a step in the environment."""
        actions = action.reshape(2, 2)
        action = {'GoalieBrain': actions[0], 'StrikerBrain': actions[1]}
        #action = {'GoalieBrain': [-1, -1], 'StrikerBrain': [6, 6]}
        #print(action)
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
