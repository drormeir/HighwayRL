import json

from env_wrapper import EnvWrapper
from random_agent import RandomAgent
from dqn_agent import HighwayAgentDQN
import os
from highway_training import *
import matplotlib.pyplot as plt


def sanity_check():
    movie_name = os.path.join(os.getcwd(), 'movie.gif')
    env_high = EnvWrapper(env_name='highway-fast-v0', config_name='highway-config/config_ex1.txt',
                          override_config=None, movie_name=movie_name)
    for i in range(1):
        print(f"Running random episode {i + 1} out of 10")
        env_high.single_episode_random_action(display=False)
    print("done")

    random_agent = RandomAgent(env_high.action_size)
    env_high.single_episode_test(random_agent, display=False, print_every_step=1)
    env_high.single_episode_training(random_agent, print_every_step=1)
    agent = HighwayAgentDQN(state_size=env_high.state_size, action_size=env_high.action_size, pytorch_device='gpu')
    agent.local_q_network.save('stam.pth')
    agent.local_q_network.load('stam.pth')


def find_hyper_parameters():
    movie_name = os.path.join(os.getcwd(), 'movie.gif')
    env_high = EnvWrapper(env_name='highway-fast-v0', config_name='highway-config/config_ex1.txt',
                          override_config=None, movie_name=movie_name)
    find_best_lr(env_high, max_episodes=100)
    find_best_gamma(env_high, max_episodes=170)
    find_best_eps_decay(env_high, max_episodes=150)
    find_best_train_every_step(env_high, max_episodes=300)
    find_best_tau(env_high, max_episodes=190)
    find_best_mini_batch(env_high, mini_batches=list(range(8, 49, 8)), max_episodes=300)
    find_best_ch_conv1(env_high, max_episodes=300)
    find_best_reward_power(env_high, max_episodes=300)
    find_best_fc2_out(env_high, max_episodes=300)


def full_training():
    movie_name = os.path.join(os.getcwd(), 'movie.gif')
    env_names = ['highway-fast-v0', 'merge-v0', 'roundabout-v0']
    base_config_dir = 'highway-config'
    config_names = [os.path.join(base_config_dir, 'config_ex1.txt'), os.path.join(base_config_dir, 'config_ex2.txt'),
                    os.path.join(base_config_dir, 'config_ex3.txt')]
    main_params = [(env_names[0], config_names[0], 'EX1'), (env_names[0], config_names[1], 'EX2'),
                   (env_names, config_names[2], 'EX3')]
    for env_name, config_name, ex_name in main_params:
        base_dir = os.path.join('results', ex_name)
        os.makedirs(base_dir, exist_ok=True)
        env = EnvWrapper(env_name=env_name, config_name=config_name, override_config=None, movie_name=movie_name)
        highway_agent = HighwayAgentDQN(state_size=env.state_size, action_size=env.action_size, pytorch_device='gpu')
        highway_agent.save_param_dict(os.path.join(base_dir, 'agent_params_dict.json'))
        env.multi_episode_train(highway_agent, print_every_episode=1, max_episodes=5000,
                                save_test_high_score_filename=os.path.join(base_dir, 'best_testing.pth'),
                                save_train_high_score_filename=os.path.join(base_dir, 'best_training.pth'),
                                save_results_filename=os.path.join(base_dir, 'env_train_results.json'))


def plot_graphs():
    for ex_name in ['EX1', 'EX2', 'EX3']:
        base_dir = os.path.join('results', ex_name)
        save_results_filename = os.path.join(base_dir, 'env_train_results.json')
        with open(save_results_filename, 'r') as f:
            results_dict = json.load(f)
            env_name = results_dict['env_name']
            config_name = results_dict['config_name']
            scores = results_dict['scores']
            lengths = results_dict['lengths']
            evaluations = results_dict['evaluations']
            q1 = []
            q2 = []
            q3 = []
            num_scores = 100
            indexes = list(range(1, len(scores)+1))
            for ind in indexes:
                if ind < num_scores:
                    score = scores[:ind]
                else:
                    score = scores[(ind-num_scores):ind]
                score_q1, score_q2, score_q3 = np.quantile(score, [0.25, 0.5, 0.75])
                q1.append(score_q1)
                q2.append(score_q2)
                q3.append(score_q3)

            fig, ax = plt.subplots(figsize=(7, 7))
            plt.title(f'Results of {ex_name}\nFor environment: {env_name}\nwith configuration {config_name}')
            plt.plot(indexes, scores, label='Episode score')
            plt.plot(indexes, q1, label=f'{num_scores} episodes 1st quartile scores')
            plt.plot(indexes, q2, label=f'{num_scores} episodes median scores')
            plt.plot(indexes, q3, label=f'{num_scores} episodes 3rd quartile scores')
            ax.set_ylabel("Episode total score")
            ax.set_xlabel("Number of training episodes")
            plt.legend(loc="upper left", prop={'size': 6})
            plt.show()


if __name__ == '__main__':
    sanity_check()
    find_hyper_parameters()
    full_training()
    plot_graphs()
