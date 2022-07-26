import numpy as np
# from env_wrapper import EnvWrapper
from dqn_agent import HighwayAgentDQN
import gc


def find_best_lr(env, learning_rates=None, max_episodes=200, print_every_episode=10):
    if learning_rates is None:
        learning_rates = list(np.linspace(1e-4, 9e-4, 9)) \
                         + list(np.linspace(1e-3, 9e-3, 9))
    ret = []
    for lr in learning_rates:
        print(f'Running with learning rate: {lr:4.2e}')
        highway_agent = HighwayAgentDQN(state_size=env.state_size, action_size=env.action_size,
                                        lr=lr, pytorch_device='gpu')
        episodes_score, episodes_length = \
            env.multi_episode_train(highway_agent, max_episodes=max_episodes,
                                    print_every_episode=print_every_episode)
        if env.evaluations and env.evaluations['score_med']:
            ret.append((lr, len(episodes_score), np.max(env.evaluations['score_med'])))
    print(ret)
    return ret


def find_best_gamma(env, gammas=None, max_episodes=200, print_every_episode=10):
    if gammas is None:
        gammas = list(np.linspace(0.980, 0.999, 20))
    ret = []
    for gamma in gammas:
        print(f'Running with gamma: {gamma:5.3f}')
        highway_agent = HighwayAgentDQN(state_size=env.state_size, action_size=env.action_size,
                                        gamma=gamma, pytorch_device='gpu')
        episodes_score, episodes_length = \
            env.multi_episode_train(highway_agent, max_episodes=max_episodes,
                                    print_every_episode=print_every_episode)
        if env.evaluations and env.evaluations['score_med']:
            ret.append((gamma, len(episodes_score), np.max(env.evaluations['score_med'])))
    print(ret)
    return ret


def find_best_eps_decay(env, eps_decays=None, max_episodes=200, print_every_episode=10):
    if eps_decays is None:
        eps_decays = list(np.linspace(0.975, 0.995, 21))
    ret = []
    for eps_decay in eps_decays:
        print(f'Running with eps_decay: {eps_decay:5.3f}')
        highway_agent = HighwayAgentDQN(state_size=env.state_size, action_size=env.action_size,
                                        eps_greedy_decay=eps_decay,
                                        pytorch_device='gpu')
        episodes_score, episodes_length = \
            env.multi_episode_train(highway_agent, max_episodes=max_episodes,
                                    print_every_episode=print_every_episode)
        if env.evaluations and env.evaluations['score_med']:
            ret.append((eps_decay, len(episodes_score), np.max(env.evaluations['score_med'])))
    print(ret)
    return ret


def find_best_train_every_step(env, train_every_episode_steps=None, max_episodes=200, print_every_episode=10):
    if train_every_episode_steps is None:
        train_every_episode_steps = list(range(1, 6, 1))
    ret = []
    for train_every_episode_step in train_every_episode_steps:
        print(f'Running with train_every_episode_step: {train_every_episode_step}')
        highway_agent = HighwayAgentDQN(state_size=env.state_size, action_size=env.action_size,
                                        train_every_episode_steps=train_every_episode_step,
                                        pytorch_device='gpu')
        episodes_score, episodes_length = \
            env.multi_episode_train(highway_agent, max_episodes=max_episodes,
                                    print_every_episode=print_every_episode)
        ret.append((train_every_episode_step, len(episodes_score), np.max(env.evaluations['score_med'])))
    print(ret)
    return ret


def find_best_tau(env, taus=None, max_episodes=200, print_every_episode=10):
    if taus is None:
        taus = list(np.linspace(0.001, 0.005, 5))
    ret = []
    for tau in taus:
        print(f'Running with tau: {tau}')
        highway_agent = HighwayAgentDQN(state_size=env.state_size, action_size=env.action_size,
                                        local_tau_weight=tau,
                                        pytorch_device='gpu')
        episodes_score, episodes_length = \
            env.multi_episode_train(highway_agent, max_episodes=max_episodes,
                                    print_every_episode=print_every_episode)
        if env.evaluations and env.evaluations['score_med']:
            ret.append((tau, len(episodes_score), np.max(env.evaluations['score_med'])))
    print(ret)
    return ret


def find_best_mini_batch(env, mini_batches=None, max_episodes=200, print_every_episode=10, print_every_step=0):
    if mini_batches is None:
        mini_batches = list(range(16, 65, 4))
    ret = []
    for mini_batch in mini_batches:
        print(f'Running with mini_batch: {mini_batch}')
        gc.collect()
        highway_agent = HighwayAgentDQN(state_size=env.state_size, action_size=env.action_size,
                                        replay_batch_size=mini_batch,
                                        pytorch_device='gpu')
        episodes_score, episodes_length = \
            env.multi_episode_train(highway_agent, max_episodes=max_episodes,
                                    print_every_episode=print_every_episode,
                                    print_every_step=print_every_step)
        if env.evaluations and env.evaluations['score_med']:
            ret.append((mini_batch, len(episodes_score), np.max(env.evaluations['score_med'])))
    print(ret)
    return ret


def find_best_ch_conv1(env, ch_conv1s=None, max_episodes=200, print_every_episode=10, print_every_step=0):
    if ch_conv1s is None:
        ch_conv1s = list(range(8, 65, 8))
    ret = []
    for ch_conv1 in ch_conv1s:
        print(f'Running with ch_conv1: {ch_conv1}')
        gc.collect()
        highway_agent = HighwayAgentDQN(state_size=env.state_size, action_size=env.action_size,
                                        ch_conv1=ch_conv1,
                                        pytorch_device='gpu')
        episodes_score, episodes_length = \
            env.multi_episode_train(highway_agent, max_episodes=max_episodes,
                                    print_every_episode=print_every_episode,
                                    print_every_step=print_every_step)
        if env.evaluations and env.evaluations['score_med']:
            ret.append((ch_conv1, len(episodes_score), np.max(env.evaluations['score_med'])))
    print(ret)
    return ret


def find_best_ch_conv2(env, ch_conv2s=None, max_episodes=200, print_every_episode=10, print_every_step=0):
    if ch_conv2s is None:
        ch_conv2s = list(range(16, 129, 8))
    ret = []
    for ch_conv2 in ch_conv2s:
        print(f'Running with ch_conv2: {ch_conv2}')
        gc.collect()
        highway_agent = HighwayAgentDQN(state_size=env.state_size, action_size=env.action_size,
                                        ch_conv2=ch_conv2,
                                        pytorch_device='gpu')
        episodes_score, episodes_length = \
            env.multi_episode_train(highway_agent, max_episodes=max_episodes,
                                    print_every_episode=print_every_episode,
                                    print_every_step=print_every_step)
        if env.evaluations and env.evaluations['score_med']:
            ret.append((ch_conv2, len(episodes_score), np.max(env.evaluations['score_med'])))
    print(ret)
    return ret


def find_best_fc0_out(env, fc0_outs=None, max_episodes=200, print_every_episode=10, print_every_step=0):
    if fc0_outs is None:
        fc0_outs = list(range(90, 361, 10))
    ret = []
    for fc0_out in fc0_outs:
        print(f'Running with fc0_out: {fc0_out}')
        gc.collect()
        highway_agent = HighwayAgentDQN(state_size=env.state_size, action_size=env.action_size,
                                        fc0_out=fc0_out,
                                        pytorch_device='gpu')
        episodes_score, episodes_length = \
            env.multi_episode_train(highway_agent, max_episodes=max_episodes,
                                    print_every_episode=print_every_episode,
                                    print_every_step=print_every_step)
        if env.evaluations and env.evaluations['score_med']:
            ret.append((fc0_out, len(episodes_score), np.max(env.evaluations['score_med'])))
    print(ret)
    return ret


def find_best_fc1_out(env, fc1_outs=None, max_episodes=200, print_every_episode=10, print_every_step=0):
    if fc1_outs is None:
        fc1_outs = list(range(120, 361, 10))
    ret = []
    for fc1_out in fc1_outs:
        print(f'Running with fc1_out: {fc1_out}')
        gc.collect()
        highway_agent = HighwayAgentDQN(state_size=env.state_size, action_size=env.action_size,
                                        fc1_out=fc1_out,
                                        pytorch_device='gpu')
        episodes_score, episodes_length = \
            env.multi_episode_train(highway_agent, max_episodes=max_episodes,
                                    print_every_episode=print_every_episode,
                                    print_every_step=print_every_step)
        if env.evaluations and env.evaluations['score_med']:
            ret.append((fc1_out, len(episodes_score), np.max(env.evaluations['score_med'])))
    print(ret)
    return ret


def find_best_fc2_out(env, fc2_outs=None, max_episodes=200, print_every_episode=10, print_every_step=0):
    if fc2_outs is None:
        fc2_outs = list(range(120, 361, 10))
    ret = []
    for fc2_out in fc2_outs:
        print(f'Running with fc2_out: {fc2_out}')
        gc.collect()
        highway_agent = HighwayAgentDQN(state_size=env.state_size, action_size=env.action_size,
                                        fc2_out=fc2_out,
                                        pytorch_device='gpu')
        episodes_score, episodes_length = env.multi_episode_train(highway_agent, max_episodes=max_episodes,
                                                                  print_every_episode=print_every_episode,
                                                                  print_every_step=print_every_step)
        if env.evaluations and env.evaluations['score_med']:
            ret.append((fc2_out, len(episodes_score), np.max(env.evaluations['score_med'])))
    print(ret)
    return ret


def find_best_reward_power(env, reward_powers=None, max_episodes=200, print_every_episode=10, print_every_step=0):
    if reward_powers is None:
        reward_powers = np.linspace(1.0, 3.0, 21)
    ret = []
    for reward_power in reward_powers:
        print(f'Running with reward_power: {reward_power}')
        gc.collect()
        highway_agent = HighwayAgentDQN(state_size=env.state_size, action_size=env.action_size,
                                        reward_power=reward_power,
                                        pytorch_device='gpu')
        episodes_score, episodes_length = env.multi_episode_train(highway_agent, max_episodes=max_episodes,
                                                                  print_every_episode=print_every_episode,
                                                                  print_every_step=print_every_step)
        if env.evaluations and env.evaluations['score_med']:
            ret.append((reward_power, len(episodes_score), np.max(env.evaluations['score_med'])))
    print(ret)
    return ret
