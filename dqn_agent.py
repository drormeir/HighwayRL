import math
import json

from replay_buffer import ReplayBuffer
from dqn_model import HighwayActorModelDQN, OptimizerDQN
import numpy as np


class HighwayAgentDQN:
    def __init__(self, env,
                 seed=0,
                 eps_greedy_min=0.01, eps_greedy_decay=0.981,
                 replay_buffer_size: int = int(1e4),
                 replay_batch_size: int = 40,
                 train_every_episode_steps=2,
                 gamma=0.987,
                 local_tau_weight=0.003,
                 num_targets: int = 1,
                 lr=5e-4,
                 ch_conv1=16,
                 ch_conv2=64,
                 fc0_out=180,
                 fc1_out=180,
                 fc2_out=200,
                 reward_power=1.5,
                 pytorch_device=None,
                 verbose_level=1):

        self.eps_greedy = 1.0
        self.eps_greedy_min = eps_greedy_min
        self.eps_greedy_decay = eps_greedy_decay
        self.action_size = env.action_size
        self.state_size = env.state_size
        self.env_name = env.env_name
        self.reward_power = reward_power
        self.random = np.random.default_rng(seed=seed)
        model_params = {'ch_conv1': ch_conv1, 'ch_conv2': ch_conv2,
                        'fc0_out': fc0_out, 'fc1_out': fc1_out, 'fc2_out': fc2_out}
        self.local_q_network = HighwayActorModelDQN(state_size=self.state_size, action_size=self.action_size,
                                                    model_params=model_params,
                                                    seed=seed, pytorch_device=pytorch_device)
        self.target_q_network = [
            HighwayActorModelDQN(state_size=self.state_size, action_size=self.action_size, model_params=model_params,
                                 seed=seed + i_target, pytorch_device=pytorch_device)
            for i_target in range(num_targets)]

        replay_buffer_params = {'sample_every': train_every_episode_steps,
                                'buffer_size': replay_buffer_size, 'batch_size': replay_batch_size}
        self.replay_buffer = ReplayBuffer(state_size=self.state_size, state_type=np.uint8,
                                          action_size=self.action_size, action_type=int,
                                          params=replay_buffer_params,
                                          seed=seed)
        self.optimizer = OptimizerDQN(self.local_q_network, lr=lr)
        self.ind_next_step_in_episode = 0
        self.gamma = gamma
        self.local_tau_weight = local_tau_weight
        self.verbose_level = verbose_level
        if self.verbose_level > 0:
            print('Initializing HighwayAgentDQN with:')
            print(f'Epsilon Greedy: min={self.eps_greedy_min} decay={self.eps_greedy_decay}')
            print(f'Gamma = {self.gamma}')
            print(f'tau = {self.local_tau_weight}')
            print(f'Replay Buffer: {replay_buffer_params}')
            print(f'Learning rate = {lr:4.2e}')
            print(f'Model params = {model_params}')
        self.params_dict = {'replay_buffer': replay_buffer_params, 'Q_network': model_params, 'gamma': gamma,
                            'tau': self.local_tau_weight, 'learning_rate': lr, 'epsilon_min': self.eps_greedy_min,
                            'epsilon_decay': self.eps_greedy_decay, 'reward_power': self.reward_power, 'seed': seed}

    def act(self, state, training):
        """Returns actions for given state as per current policy.

        Params
        ======
            state (array_like): current state
            training (bool): true = epsilon-greedy action selection
        """
        if training:
            if self.random.random() < self.eps_greedy:
                # explore the environment using random move
                return int(self.random.choice(self.action_size))
        # exploit the agent knowledge
        return self.local_q_network.act(state)

    def step(self, state, action, reward, next_state, done, info):
        # ind_curr_step_in_episode = self.ind_next_step_in_episode
        if done:
            # episode can end because reaching maximum steps, hence set done == True iff crash
            done = info['crashed']
            # preparing for next episode...
            self.ind_next_step_in_episode = 0
            self.eps_greedy = max(self.eps_greedy * self.eps_greedy_decay, self.eps_greedy_min)
        else:
            self.ind_next_step_in_episode += 1
        # minimal velocity has reward of 0.7 and maximal velocity has reward close to 1.0
        # hence, encourage agent achieve higher velocity
        reward = math.pow(reward, self.reward_power)
        state = state.astype(np.uint8)
        next_state = next_state.astype(np.uint8)
        experiences = self.replay_buffer.add_and_sample(state, action, reward, next_state, done)
        if experiences is None:
            return
        states, actions, rewards, next_states, dones = experiences

        q_targets = [q_target.q_targets(rewards, self.gamma, dones, next_states) for q_target in self.target_q_network]

        q_expected = self.local_q_network.q_expected(states, actions)

        self.optimizer.step(q_expected, q_targets)

        for q_target in self.target_q_network:
            q_target.soft_update_from_local(self.local_q_network, local_tau_weight=self.local_tau_weight)

    def save_param_dict(self, filename):
        with open(filename, 'w') as f:
            json.dump(self.params_dict, f)

    def load_param_dict(self, filename):
        with open(filename, 'r') as f:
            self.params_dict = json.load(f)
