import numpy as np


class RandomAgent:
    def __init__(self, num_actions, seed=0):
        self.random = np.random.default_rng(seed=seed)
        self.num_actions = num_actions

    def act(self, state, training=None):
        if state is None or training is None:
            return -1
        return int(self.random.choice(self.num_actions))

    def step(self, state, action, reward, next_state, done, info):
        if state is None or action is None or reward is None or next_state is None or done is None or info is None:
            print(f"None in RandomAgent.step()  num_actions={self.num_actions}")

