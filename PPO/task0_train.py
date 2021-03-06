import sys, os

curr_path = os.path.dirname(__file__)
parent_path = os.path.dirname(curr_path)
sys.path.append(parent_path)  # add current terminal path to sys.path

import gym
import torch
import datetime
from PPO.agent_false import PPOAgent
from common.plot import plot_rewards
from common.utils import save_results, make_dir

curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")  # obtain current time


class PPOConfig:
    def __init__(self) -> None:
        self.env = 'CartPole-v0'
        self.algo = 'PPO'
        self.result_path = curr_path + "/results/" + self.env + '/' + curr_time + '/results/'  # path to save results
        self.model_path = curr_path + "/results/" + self.env + '/' + curr_time + '/models/'  # path to save models
        self.train_eps = 1000  # max training episodes
        self.eval_eps = 50
        self.batch_size = 1000
        self.gamma = 0.99
        self.n_epoch = 4
        self.lr = 0.0003
        #self.critic_lr = 0.0003
        self.gae_lambda = 0.95
        self.clip_param = 0.2
        self.hidden_dim = 256
        self.capacity = 100000

        self.update_fre = 20  # frequency of agent update
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # check gpu


def env_agent_config(cfg, seed=1):
    env = gym.make(cfg.env)
    env.seed(seed)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = PPOAgent(state_dim, action_dim, cfg)
    return env, agent


def train(cfg, env, agent):
    print('Start to train !')
    print(f'Env:{cfg.env}, Algorithm:{cfg.algo}, Device:{cfg.device}')
    rewards = []
    ma_rewards = []  # moving average rewards
    running_steps = 0
    for i_ep in range(cfg.train_eps):
        state = env.reset()
        done = False
        ep_reward = 0
        epi_states = []
        epi_actions = []
        epi_rewards = []
        while not done:
            """action, prob, val = agent.choose_action(state)
            state_, reward, done, _ = env.step(action)
            running_steps += 1
            ep_reward += reward
            agent.memory.push(state, action, prob, val, reward, done)
            if running_steps % cfg.update_fre == 0:
                agent.update()
            state = state_"""
            action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            epi_states.append(state)
            epi_actions.append(action)
            epi_rewards.append(reward)
            state = next_state
            running_steps += 1
            ep_reward += reward
        rewards.append(ep_reward)
        if ma_rewards:
            ma_rewards.append(
                0.9 * ma_rewards[-1] + 0.1 * ep_reward)
        else:
            ma_rewards.append(ep_reward)
        #process samples
        old_log_probs, gaes, returns = agent.process_samples(epi_states, epi_actions, epi_rewards)
        for i in range(len(epi_states)):
            agent.memory.push((epi_states[i], epi_actions[i], old_log_probs[i], gaes[i], returns[i]))

        #update
        if len(agent.memory) < agent.batch_size:
            continue
        #print("staring training")
        if (i_ep+1) % 3 == 0:
            agent.update()


        print(f"Episode:{i_ep + 1}/{cfg.train_eps}, Reward:{ep_reward:.3f}")
    print('Complete training???')
    return rewards, ma_rewards


def eval(cfg, env, agent):
    print('Start to eval !')
    print(f'Env:{cfg.env}, Algorithm:{cfg.algo}, Device:{cfg.device}')
    rewards = []
    ma_rewards = []  # moving average rewards
    for i_ep in range(cfg.eval_eps):
        state = env.reset()
        done = False
        ep_reward = 0
        while not done:
            action = agent.choose_action(state)
            state_, reward, done, _ = env.step(action)
            ep_reward += reward
            state = state_
        rewards.append(ep_reward)
        if ma_rewards:
            ma_rewards.append(
                0.9 * ma_rewards[-1] + 0.1 * ep_reward)
        else:
            ma_rewards.append(ep_reward)
        print(f"Episode:{i_ep + 1}/{cfg.train_eps}, Reward:{ep_reward:.3f}")
    print('Complete evaling???')
    return rewards, ma_rewards


def show(env, agent):
    state = env.reset()
    while True:
        env.render()
        action = agent.choose_action(state)
        next_state, reward, done, _ = env.step(action)
        state = next_state
        if done:
            break


if __name__ == '__main__':
    cfg = PPOConfig()
    # train
    env, agent = env_agent_config(cfg, seed=1)
    rewards, ma_rewards = train(cfg, env, agent)
    make_dir(cfg.result_path, cfg.model_path)
    agent.save(path=cfg.model_path)
    save_results(rewards, ma_rewards, tag='train', path=cfg.result_path)
    plot_rewards(rewards, ma_rewards, tag="train",
                 algo=cfg.algo, path=cfg.result_path)
    # eval
    env, agent = env_agent_config(cfg, seed=10)
    agent.load(path=cfg.model_path)
    rewards, ma_rewards = eval(cfg, env, agent)
    save_results(rewards, ma_rewards, tag='eval', path=cfg.result_path)
    plot_rewards(rewards, ma_rewards, tag="eval", env=cfg.env, algo=cfg.algo, path=cfg.result_path)

    show(env, agent)