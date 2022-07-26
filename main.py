from env_wrapper import EnvWrapper
from random_agent import RandomAgent
from dqn_agent import HighwayAgentDQN
import os
from highway_training import *

if __name__ == '__main__':
    movie_name = os.path.join(os.getcwd(), 'movie.gif')
    env = EnvWrapper(env_name='highway-fast-v0', config_name='highway-config/config_ex1.txt',
                     override_config=None, movie_name=movie_name)
    for i in range(1):
        print(f"Running random episode {i + 1} out of 10")
        env.single_episode_random_action(display=False)
    print("done")

    random_agent = RandomAgent(env.action_size)
    env.single_episode_test(random_agent, display=False, print_every_step=1)
    env.single_episode_training(random_agent, print_every_step=1)

    # find_best_lr(env, max_episodes=100)
    # find_best_gamma(env, max_episodes=170)
    # find_best_eps_decay(env, max_episodes=150)
    # find_best_train_every_step(env, max_episodes=300)
    # find_best_tau(env, max_episodes=190)
    # find_best_mini_batch(env, mini_batches=list(range(8, 49, 8)), max_episodes=300)
    # find_best_ch_conv1(env, max_episodes=300)
    # find_best_reward_power(env, max_episodes=300)
    # find_best_fc2_out(env, max_episodes=300)
    base_config_dir = 'highway-config'
    env_names = ['highway-fast-v0', 'merge-v0', 'roundabout-v0']
    config_names = [os.path.join(base_config_dir, 'config_ex1.txt'), os.path.join(base_config_dir, 'config_ex2.txt'),
                    os.path.join(base_config_dir, 'config_ex3.txt')]
    main_params = [(env_names[0], config_names[0], 'EX1'), (env_names[0], config_names[1], 'EX2'),
                   (env_names, config_names[2], 'EX3')]
    for env_name, config_name, ex_name in main_params:
        base_dir = os.path.join('results', ex_name)
        os.makedirs(base_dir, exist_ok=True)
        env = EnvWrapper(env_name=env_name, config_name=config_name, override_config=None, movie_name=movie_name)
        highway_agent = HighwayAgentDQN(env, pytorch_device='gpu')
        highway_agent.save_param_dict(os.path.join(base_dir, 'agent_params_dict.json'))
        env.multi_episode_train(highway_agent, print_every_episode=1, max_episodes=5000,
                                save_test_high_score_filename=os.path.join(base_dir, 'best_testing.pth'),
                                save_train_high_score_filename=os.path.join(base_dir, 'best_training.pth'),
                                save_results_filename=os.path.join(base_dir, 'env_train_results.json'))
