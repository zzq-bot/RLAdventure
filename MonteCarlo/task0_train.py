import sys,os
curr_path = os.path.dirname(__file__)
parent_path = os.path.dirname(curr_path)
sys.path.append(parent_path)
import torch
import datetime

from common.utils import save_results,make_dir
from common.plot import plot_rewards
from MonteCarlo.agent import FirstVisitMC
from envs.racetrack_env import RacetrackEnv

curr_time = datetime.datetime.now().strftime(
    "%Y%m%d-%H%M%S")


class MCConfig:
    def __init__(self):
        self.algo = "MC"
        self.env = "Racetrack"
        self.result_path = curr_path + "/outputs/" + self.env + \
                           '/' + curr_time + '/results/'  # path to save results
        self.model_path = curr_path + "/outputs/" + self.env + \
                          '/' + curr_time + '/models/'  # path to save models

        self.eps = 0.15
        self.gamma = 0.9  # gamma: Gamma discount factor.
        self.train_eps = 200
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")  # check gpu


def env_agent_config(config,seed=1):
    env = RacetrackEnv()
    action_dim = 9
    agent = FirstVisitMC(action_dim,config)
    return env, agent


def train(cfg, env, agent):
    print('Start to training !')
    print(f'Env:{cfg.env}, Algorithm:{cfg.algo}, Device:{cfg.device}')
    rewards = []
    average_rewards = []
    for i_ep in range(cfg.train_eps):
        state = env.reset()
        ep_reward = 0
        one_ep_experience = []
        while True:
            action = agent.choose_action(state)
            nxt_state, reward, done = env.step(action)
            ep_reward += reward
            one_ep_experience.append((state, action, reward))
            state = nxt_state
            if done:
                break
        rewards.append(ep_reward)
        if average_rewards:
            average_rewards.append(average_rewards[-1]*0.9+ep_reward*0.1)
        else:
            average_rewards.append(ep_reward)
        agent.update(one_ep_experience)
        if (i_ep+1)%10 == 0:
            print(f"Episode:{i_ep + 1}/{cfg.train_eps}: Reward:{ep_reward}")
    print("finish training")
    return rewards, average_rewards


def eval(cfg, env, agent):
    print('Start to eval !')
    print(f'Env:{cfg.env}, Algorithm:{cfg.algo}, Device:{cfg.device}')
    rewards = []
    average_rewards = []
    for i_ep in range(cfg.train_eps):
        state = env.reset()
        ep_reward = 0
        while True:
            action = agent.choose_action(state)
            nxt_state, reward, done = env.step(action)
            ep_reward += reward
            state = nxt_state
            if done:
                break
        rewards.append(ep_reward)
        if average_rewards:
            average_rewards.append(average_rewards[-1] * 0.9 + ep_reward*0.1)
        else:
            average_rewards.append(ep_reward)
        if (i_ep + 1) % 10 == 0:
            print(f"Episode:{i_ep + 1}/{cfg.train_eps}: Reward:{ep_reward}")
    return rewards, average_rewards

if __name__=="__main__":
    cfg = MCConfig()

    '''train'''
    env, agent = env_agent_config(cfg)
    rewards, ma_rewards = train(cfg, env, agent)
    make_dir(cfg.result_path, cfg.model_path)
    agent.save(path=cfg.model_path)
    save_results(rewards, ma_rewards, tag='train', path=cfg.result_path)
    plot_rewards(rewards, ma_rewards, tag="train",
                 algo=cfg.algo, path=cfg.result_path)
    ''' eval '''
    env, agent = env_agent_config(cfg, seed=10)
    agent.load(path=cfg.model_path)
    rewards, ma_rewards = eval(cfg, env, agent)
    save_results(rewards, ma_rewards, tag='eval', path=cfg.result_path)
    plot_rewards(rewards, ma_rewards, tag="eval", env=cfg.env, algo=cfg.algo, path=cfg.result_path)