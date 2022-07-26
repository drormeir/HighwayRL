import json

import numpy as np
import gym
import pprint
import highway_env  # registration of environment factory
from highway_env.envs.common.action import Action
import ast
import imageio
import os

gym.logger.set_level(gym.logger.ERROR)


class EnvWrapper:
    def __init__(self, env_name, config_name=None, override_config=None, display=False, seed=0, stochastic=0.15,
                 max_steps=500, movie_name=None, evaluation_num_episodes=20, evaluate_every_steps=2000, verbose=True):
        self.save_train_high_score_filename = None
        self.save_test_high_score_filename = None
        self.evaluation_num_episodes = evaluation_num_episodes
        self.evaluations = None
        self.display = display
        self.random = None
        self.movie = None
        self.stochastic = stochastic
        self.movie_name = movie_name
        self.episode_score = 0.0
        self.total_training_steps = 0
        self.episode_num_steps = 0
        self.print_every_step = 0
        self.evaluate_every_steps = evaluate_every_steps
        self.save_seed = 0
        self.max_steps = max_steps
        self.evaluation_high_score = None
        if verbose:
            print('Generating GYM environment:', env_name)
        self.env_name = env_name
        if isinstance(env_name, list):
            self.env = [gym.make(name) for name in env_name]
        else:
            self.env = [gym.make(env_name)]
        self.ind_env = -1
        self.config_name = config_name
        self.override_config = override_config
        self.seed(seed)
        self.test_env = None
        if override_config is None:
            override_config = {}
        if config_name:
            if verbose:
                print('Reading configuration file:', config_name)
            with open(config_name, 'r') as f:
                contents = f.read()
                if verbose:
                    print('Content of configuration file:\n', contents)
                config1 = ast.literal_eval(contents)
                f.close()
        else:
            config1 = {}

        for key, value in override_config.items():
            config1[key] = value
        if config1:
            if verbose:
                pprint.pprint(config1)
            for env in self.env:
                env.configure(config1)
        env = self.env[0]
        observation = env.reset()
        self.action_names = {}
        for key, value in env.action_type.actions_indexes.items():
            self.action_names[value] = key
        self.action_size = len(self.action_names)
        self.state_size = env.observation_space.shape
        if verbose:
            print("=" * 80)
            print('Observation object type:', type(observation))
            print('Observation datatype:   ', observation.dtype)
            print('Observation shape:      ', observation.shape)
            print('Observation space:      ', env.observation_space)
            print('Action space type:      ', env.action_space)
            print('Action space values:    ', env.action_type.actions_indexes)
            print('Environment full configuration:')
            pprint.pprint(env.config)

    def multi_episode_train(self, agent, max_episodes=10000, print_every_episode=None, seed=0,
                            stop_on_first_success=False, print_every_step=0, break_high_score=100,
                            save_test_high_score_filename=None, save_train_high_score_filename=None,
                            save_results_filename=None):
        self.ind_env = -1
        if seed is not None:
            self.seed(seed)
        if save_test_high_score_filename is not None:
            self.save_test_high_score_filename = save_test_high_score_filename
        if save_train_high_score_filename is not None:
            self.save_train_high_score_filename = save_train_high_score_filename
        if print_every_episode is None:
            print_every_episode = int(print_every_step > 0)
        if print_every_episode > 0:
            print(f'Starting multi episode training...')
        self.reset_evaluations()
        episodes_score = []
        episodes_length = []
        high_score = [-np.inf] * 5
        available_episodes = break_high_score * 2
        for _ in range(max_episodes):
            available_episodes -= 1
            eps_greedy = agent.eps_greedy
            success, new_evaluation_high_score = self.single_episode_training(agent, print_every_step=print_every_step)
            episodes_score.append(self.episode_score)
            episodes_length.append(self.episode_num_steps)
            last_recent_scores = min(len(episodes_score), break_high_score)
            score_q1, score_q2, score_q3 = np.quantile(episodes_score[-last_recent_scores:], [0.25, 0.5, 0.75])
            length_q1, length_q2, length_q3 = np.quantile(episodes_length[-last_recent_scores:], [0.25, 0.5, 0.75])
            if self.save_train_high_score_filename is not None and score_q2 > high_score[2]:
                agent.local_q_network.save(self.save_train_high_score_filename)
            new_train_high_score = False
            for ind, val in enumerate([episodes_score[-1], score_q1, score_q2, length_q1, length_q2]):
                if val > high_score[ind]:
                    high_score[ind] = val
                    new_train_high_score = True
            if new_evaluation_high_score or new_train_high_score:
                available_episodes = break_high_score
            if print_every_episode > 0:
                if success or new_evaluation_high_score or new_train_high_score\
                        or len(episodes_score) % print_every_episode == 0 \
                        or len(episodes_score) == max_episodes:
                    if new_train_high_score:
                        print('New training high score')
                    if new_evaluation_high_score:
                        print('New evaluation high score')
                    print(f'episode: {len(episodes_score):4}',
                          f'eps_greedy:{eps_greedy:6.4f}',
                          f'length:{self.episode_num_steps:4}',
                          'score:', str_episode_score(episodes_score[-1]),
                          'avg scores:', str_episode_score(score_q1), str_episode_score(score_q2),
                          str_episode_score(score_q3))
            if stop_on_first_success and success:
                if print_every_episode > 0:
                    print('Found successful episode --> finish training')
                break
            if available_episodes < 1:
                if print_every_episode > 0:
                    print('Could not train any further.')
                break
            pass  # end training loop
        if print_every_episode > 0:
            str_out = 'Multi episode training is over with max score: ' + str_episode_score(np.max(episodes_score)) +\
                      f' after {np.argmax(episodes_score) + 1} episodes'
            if self.evaluations:
                evaluations_med = self.evaluations['score_med']
                if evaluations_med:
                    str_out = str_out + ' Max evaluation: ' + str_episode_score(np.max(evaluations_med))
            print(str_out)
        if save_results_filename is not None:
            results_dict = {'env_name': self.env_name, 'config_name': self.config_name, 'scores': episodes_score,
                            'lengths': episodes_length, 'evaluations': self.evaluations}
            with open(save_results_filename, 'w') as f:
                json.dump(results_dict, f)

        return episodes_score, episodes_length

    def single_episode_training(self, agent, print_every_step=None):
        if print_every_step is None:
            print_every_step = self.print_every_step
        if print_every_step > 0:
            print(f'Starting single episode training...')
        state, info = self.reset(display=False)
        done = False
        new_evaluation_high_score = False
        while not done:
            action = agent.act(state, training=True)  # select an action
            next_state, reward, done, info = self.step(action)
            if print_every_step > 0:
                if self.episode_num_steps % print_every_step == 0:
                    print(f'step: {self.episode_num_steps:4}, action: {self.action_names[action] + ",":11}',
                          'reward:', str_reward(reward), 'done:', str_done(done), 'info:', info)
            agent.step(state, action, reward, next_state, done, info)
            self.total_training_steps += 1
            if self.evaluate_every_steps > 0 and self.total_training_steps % self.evaluate_every_steps == 0:
                if self.evaluate_agent(agent, print_every_episode=100):
                    new_evaluation_high_score = True
            state = next_state
        if print_every_step > 0:
            print(f'Single episode training is over after {self.episode_num_steps} steps',
                  f'with total score: {self.episode_score}')
        return not info['crashed'], new_evaluation_high_score

    def single_episode_random_action(self, display=False):
        print(f'starting random episode...')
        state, info = self.reset(display=display)
        done = False
        while not done:
            action = int(self.env[self.ind_env].action_space.sample())
            state, reward, done, info = self.step(action)
            print(f'step: {self.episode_num_steps:3}, action: {self.action_names[action] + ",":11}',
                  'reward:', str_reward(reward), 'done:', str_done(done), 'total score:',
                  str_episode_score(self.episode_score))
        print('Episode ended with average reward:', str_reward(self.episode_score / self.episode_num_steps))
        return not info['crashed']

    def reset_evaluations(self):
        self.total_training_steps = 0
        self.evaluations = {'total_training_steps': [],
                            'score_med': [], 'score_q1': [], 'score_q3': [],
                            'length_med': [], 'length_q1': [], 'length_q3': []}
        self.evaluation_high_score = [-np.inf] * 4

    def evaluate_agent(self, agent, print_every_episode=0):
        if print_every_episode > 0:
            print(f'Evaluation phase after {self.total_training_steps} training steps'
                  f' using {self.evaluation_num_episodes} episodes...')
        if self.test_env is None:
            self.test_env = EnvWrapper(env_name=self.env_name, config_name=self.config_name,
                                       override_config=self.override_config, display=False, seed=self.save_seed,
                                       stochastic=self.stochastic, max_steps=self.max_steps, movie_name=None,
                                       evaluate_every_steps=self.evaluate_every_steps,
                                       evaluation_num_episodes=self.evaluation_num_episodes,
                                       verbose=False)
            self.test_env.reset_evaluations()
        scores = []
        lengths = []
        for _ in range(self.evaluation_num_episodes):
            self.test_env.single_episode_test(agent, display=False, print_every_step=0)
            scores.append(self.test_env.episode_score)
            lengths.append(self.test_env.episode_num_steps)
        score_q1, score_q2, score_q3 = np.quantile(scores, [0.25, 0.5, 0.75])
        length_q1, length_q2, length_q3 = np.quantile(lengths, [0.25, 0.5, 0.75])
        if print_every_episode > 0:
            print('Evaluation scores =', str_episode_score(score_q1), str_episode_score(score_q2),
                  str_episode_score(score_q3),
                  'lengths =', str_episode_score(length_q1), str_episode_score(length_q2),
                  str_episode_score(length_q3))
        if self.save_test_high_score_filename is not None and score_q2 > self.evaluation_high_score[1]:
            agent.local_q_network.save(self.save_test_high_score_filename)
        evaluation_new_high_score = False
        for ind, val in enumerate([score_q1, score_q2, length_q1, length_q2]):
            if val > self.evaluation_high_score[ind]:
                self.evaluation_high_score[ind] = val
                evaluation_new_high_score = True
        self.evaluations['total_training_steps'].append(self.total_training_steps)
        self.evaluations['score_med'].append(score_q2)
        self.evaluations['score_q1'].append(score_q1)
        self.evaluations['score_q3'].append(score_q3)
        self.evaluations['length_med'].append(length_q2)
        self.evaluations['length_q1'].append(length_q1)
        self.evaluations['length_q3'].append(length_q3)
        return evaluation_new_high_score

    def single_episode_test(self, agent, display, print_every_step=None):
        if print_every_step is None:
            print_every_step = self.print_every_step
        if print_every_step > 0:
            print(f'Starting single episode test...')
        state, info = self.reset(display=display)
        done = False
        while not done:
            action = agent.act(state, training=False)  # select an action
            state, reward, done, info = self.step(action)
            if print_every_step > 0:
                if self.episode_num_steps % print_every_step == 0:
                    print(f'step: {self.episode_num_steps:4}, action: {self.action_names[action] + ",":11}',
                          'reward:', str_reward(reward), 'done: {done} info: {info}')
        if print_every_step > 0:
            print(f'Single episode test is over after {self.episode_num_steps} steps',
                  'with total score:', str_episode_score(self.episode_score))
        return not info['crashed']

    def reset(self, ind_env=None, display=None):
        if display is not None:
            self.display = display
        if self.max_steps is not None:
            for env in self.env:
                env.configure({'duration': self.max_steps})
        self.episode_score = 0.0
        self.episode_num_steps = 0
        if ind_env is not None:
            self.ind_env = ind_env
        elif self.ind_env < 0 or self.ind_env >= len(self.env) - 1:
            self.ind_env = 0
        else:
            self.ind_env += 1
        res = self.env[self.ind_env].reset()
        self.movie = [] if self.display else None
        self.render()
        info = {'crashed': True}  # pivot value
        return res, info

    def step(self, action: int):
        if self.stochastic is not None:
            if self.random.random() < self.stochastic:
                action = int(self.random.choice(list(range(action)) + list(range(action + 1, self.action_size))))
        if not isinstance(action, int) or action < 0:
            print(f'Invalid action type={type(action)} value={action}')
            assert action >= 0
        state, reward, done, info = self.env[self.ind_env].step(action)
        self.render()
        self.episode_num_steps += 1
        assert self.episode_num_steps <= self.max_steps
        self.episode_score += reward
        if done:  # done
            self.close()
        return state, reward, done, info

    def close(self):
        self.env[self.ind_env].close()
        if self.movie:
            if self.movie_name:
                try:
                    os.remove(self.movie_name)
                except OSError:
                    pass
                with imageio.get_writer(self.movie_name, mode='I') as writer:
                    for image in self.movie:
                        writer.append_data(image)
            self.movie = []

    def seed(self, seed=None):
        if seed is None:
            seed = self.save_seed
        else:
            self.save_seed = seed
        self.random = np.random.default_rng(seed=seed)
        for env in self.env:
            env.seed(seed)

    def render(self):
        if not self.display:
            return
        self.movie.append(self.env[self.ind_env].render(mode='rgb_array'))


def str_reward(reward):
    return f'{reward:5.3f}'


def str_episode_score(score):
    return f'{score:8.3f}'


def str_done(done):
    return f'{str(done):5}'
