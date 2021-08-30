import sys,os
curr_path = os.path.dirname(__file__)
parent_path = os.path.dirname(curr_path)
sys.path.append(parent_path)
import torch
import datetime
import gym

from common.utils import save_results, make_dir
from common.plot import plot_rewards
from DQN.agent import DQN

curr_time = datetime.datetime.now().strftime(
    "%Y%m%d-%H%M%S")


class DQNConfig:
    def __init__(self):
        self.algo = 'DQN'
        self.env = 'CartPole-v0'  # 0 up, 1 right, 2 down, 3 left
        self.result_path = curr_path + "/outputs/" + self.env + '/' + curr_time + '/results/'  # path to save results
        self.model_path = curr_path + "/outputs/" + self.env + '/' + curr_time + '/models/'  # path to save models
        self.train_eps = 300
        self.eval_eps = 50
        self.gamma = 0.95  # gamma: Gamma discount factor.
        self.epsilon_start = 0.90  # e-greedy策略中初始epsilon
        self.epsilon_end = 0.01  # e-greedy策略中的终止epsilon
        self.epsilon_decay = 500  # e-greedy策略中epsilon的衰减率
        self.lr = 0.0001  # learning rate: step size parameter
        self.capacity = 100000
        self.batch_size = 64
        self.target_update = 4
        self.hidden_dim = 256
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")  # check gpu


def env_agent_config(config, seed=1, mode='train'):
    env = gym.make(cfg.env)
    env.seed(seed)
    #区分连续/离散状态空间
    if isinstance(env.observation_space, gym.spaces.box.Box):
        assert len(env.observation_space.shape) == 1
        state_dim = env.observation_space.shape[0]
    elif isinstance(env.observation_space, gym.spaces.discrete.Discrete):
        state_dim = env.observation_space.n
    act_dim = env.action_space.n
    agent = DQN(state_dim, act_dim, config, mode)
    return env, agent


def train(cfg, env, agent):
    print("Start Train")
    print(f'Env:{cfg.env}, Algorithm:{cfg.algo}, Device:{cfg.device}')
    rewards = []
    ma_rewards = []
    for i_episode in range(cfg.train_eps):
        state = env.reset()
        ep_reward = 0
        while True:
            action = agent.choose_action(state)
            nxt_state, reward, done, _ = env.step(action)
            agent.memory.push([agent.process_state(state), action, reward, agent.process_state(nxt_state), done])
            ep_reward += reward
            state = nxt_state
            if done:
                break
            if len(agent.memory) < agent.batch_size:
                continue
            agent.update()
        if (i_episode+1)%cfg.target_update == 0:
            agent.target_net.load_state_dict(agent.policy_net.state_dict())
            agent.target_net.eval()
        rewards.append(ep_reward)
        if ma_rewards:
            ma_rewards.append(0.9*ma_rewards[-1]+0.1*ep_reward)
        else:
            ma_rewards.append(ep_reward)

        if (i_episode+1) % 10 == 0:
            print("Episode:{}/{}: Reward:{}".format(i_episode + 1, cfg.train_eps, ep_reward))
    return rewards, ma_rewards


def eval(cfg, env, agent):
    rewards = []
    ma_rewards = []
    for i_episode in range(cfg.eval_eps):
        state = env.reset()
        ep_reward = 0
        while True:
            action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            ep_reward += reward
            state = next_state
            if done:
                break
        if ma_rewards:
            ma_rewards.append(ma_rewards[-1]*0.9+ep_reward*0.1)
        else:
            ma_rewards.append(ep_reward)
        rewards.append(ep_reward)
        if (i_episode+1) % 10 == 0:
            print("Episode:{}/{}: Reward:{}".format(i_episode+1, cfg.eval_eps, ep_reward))
    print('Complete evaling！')
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


if __name__=="__main__":
    cfg = DQNConfig()
    env, agent = env_agent_config(cfg, seed=1, mode='train')
    rewards, ma_rewards = train(cfg, env, agent)
    make_dir(cfg.result_path, cfg.model_path)
    agent.save(path=cfg.model_path)
    save_results(rewards, ma_rewards, tag='train', path=cfg.result_path)
    plot_rewards(rewards, ma_rewards, tag='train', env=cfg.env, algo=cfg.algo, path=cfg.result_path)

    env, agent = env_agent_config(cfg, seed=10, mode='eval')
    agent.load(path=cfg.model_path)
    rewards, ma_rewards = eval(cfg, env, agent)
    save_results(rewards, ma_rewards, tag='eval', path=cfg.result_path)
    plot_rewards(rewards, ma_rewards, tag='eval', env=cfg.env, algo=cfg.algo, path=cfg.result_path)

    show(env, agent)