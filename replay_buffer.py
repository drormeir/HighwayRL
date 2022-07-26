import numpy as np


class ReplayBuffer:
    def __init__(self,
                 state_size, state_type,
                 action_size, action_type,
                 params=None,
                 seed: int = 0):
        if params is None:
            params = {'sample_every': 1, 'buffer_size': int(1e6), 'batch_size': 64}
        assert action_type == int or action_type == float
        self.state_size = state_size
        if action_type == int:  # DQN
            self.action_begin = 0
            self.action_end = action_size
            self.action_size = 1
        else:  # action_type == float --> DDPG
            self.action_begin = None
            self.action_end = None
            self.action_size = action_size
        self.state_type = state_type
        self.action_type = action_type
        self.buffer_size = params['buffer_size']
        self.batch_size = params['batch_size']
        self.current_len = 0
        self.sample_every = params['sample_every']
        self.last_sample_len = 0
        self.random = np.random.default_rng(seed=seed)

        def np_empty(buffer_size, shape, d_type):
            if isinstance(shape, int):
                shape = (shape,)
            buffer_shape = (buffer_size,) + shape
            return np.empty(buffer_shape, dtype=d_type)

        self.states = np_empty(self.buffer_size, self.state_size, d_type=self.state_type)
        self.actions = np_empty(self.buffer_size, self.action_size, d_type=self.action_type)
        self.rewards = np_empty(self.buffer_size, 1, d_type=np.float32)
        self.next_states = np_empty(self.buffer_size, self.state_size, d_type=self.state_type)
        self.dones = np_empty(self.buffer_size, 1, d_type=int)

        self.res_states = np_empty(self.batch_size, self.state_size, d_type=self.state_type)
        self.res_actions = np_empty(self.batch_size, self.action_size, d_type=self.action_type)
        self.res_rewards = np_empty(self.batch_size, 1, d_type=np.float32)
        self.res_next_states = np_empty(self.batch_size, self.state_size, d_type=self.state_type)
        self.res_dones = np_empty(self.batch_size, 1, d_type=int)

    def add_and_sample(self, state, action, reward, next_state, done):
        self.add(state, action, reward, next_state, done)
        if self.current_len < self.last_sample_len + self.sample_every:
            # No new mini batch
            return None
        self.last_sample_len = self.current_len
        return self.sample()

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        active_len = min(self.current_len, self.buffer_size)
        if active_len < self.batch_size:
            # Not enough samples are available in memory
            return None
        indexes = self.random.choice(range(active_len), size=self.batch_size, replace=False)
        self.res_states[:] = self.states[indexes, :]
        self.res_actions[:] = self.actions[indexes, :]
        self.res_rewards[:] = self.rewards[indexes, :]
        self.res_next_states[:] = self.next_states[indexes, :]
        self.res_dones[:] = self.dones[indexes, :]
        return self.res_states, self.res_actions, self.res_rewards, self.res_next_states, self.res_dones

    def add(self, state, action, reward, next_state, done):
        if self.action_type == int and self.action_size == 1:
            assert action >= self.action_begin
            assert action < self.action_end
        ind_pos = self.current_len % self.buffer_size
        self.states[ind_pos, :] = state
        self.actions[ind_pos][0] = action
        self.rewards[ind_pos][0] = reward
        self.next_states[ind_pos, :] = next_state
        self.dones[ind_pos][0] = done
        self.current_len += 1
