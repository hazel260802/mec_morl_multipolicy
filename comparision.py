import gym as gym
import torch
import numpy as np
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import tianshou as ts
from copy import deepcopy
from tianshou.env import DummyVectorEnv
from torch.optim.lr_scheduler import LambdaLR
import torch.nn.functional as F
import os
import time
import json
import math
from tqdm import tqdm
import matplotlib.pyplot as plt
from env import SDN_Env
from network import conv_mlp_net
from tianshou.utils.net.discrete import Actor, Critic
from paretoset import paretoset
import pandas as pd

cloud_num = 1
edge_num = 1
expn = 'exp1'
config = 'multi-edge'
lr, epoch, batch_size = 1e-6, 1, 1024 * 4
train_num, test_num = 64, 1024
gamma, lr_decay = 0.9, None
buffer_size = 100000
eps_train, eps_test = 0.1, 0.00
step_per_epoch, episode_per_collect = 100 * train_num * 700, train_num
writer = SummaryWriter('tensor-board-log/ppo')
logger = ts.utils.TensorboardLogger(writer)
is_gpu_default = torch.cuda.is_available()

gae_lambda, max_grad_norm = 0.95, 0.5
vf_coef, ent_coef = 0.5, 0.0
rew_norm, action_scaling = False, False
bound_action_method = "clip"
eps_clip, value_clip = 0.2, False
repeat_per_collect = 2
dual_clip, norm_adv = None, 0.0
recompute_adv = 0

INPUT_CH = 67
FEATURE_CH = 512
MLP_CH = 1024

class sdn_net(nn.Module):
    def __init__(self, mode='actor', is_gpu=is_gpu_default):
        super().__init__()
        self.is_gpu = is_gpu
        self.mode = mode
        if self.mode == 'actor':
            self.network = conv_mlp_net(conv_in=INPUT_CH, conv_ch=FEATURE_CH, mlp_in=(edge_num+cloud_num)*FEATURE_CH,\
                                    mlp_ch=MLP_CH, out_ch=edge_num+cloud_num, block_num=3)
        else:
            self.network = conv_mlp_net(conv_in=INPUT_CH, conv_ch=FEATURE_CH, mlp_in=(edge_num+cloud_num)*FEATURE_CH,\
                                    mlp_ch=MLP_CH, out_ch=cloud_num, block_num=3)

    def load_model(self, filename):
        map_location = lambda storage, loc: storage
        self.load_state_dict(torch.load(filename, map_location=map_location))
        print('load model!')

    def save_model(self, filename):
        directory = os.path.dirname(filename)
        os.makedirs(directory, exist_ok=True)
        torch.save(self.state_dict(), filename)

    def forward(self, obs, state=None, info={}):
        state = torch.tensor(obs).float().clone().detach().requires_grad_(True)
        if self.is_gpu:
            state = state.cuda()
        logits = self.network(state)
        return logits, state

class Actor(nn.Module):
    def __init__(self, is_gpu=is_gpu_default, dist_fn=None):
        super().__init__()
        self.is_gpu = is_gpu
        self.net = sdn_net(mode='actor')
        self.dist_fn = dist_fn  

    def load_model(self, filename):
        map_location = lambda storage, loc: storage
        self.load_state_dict(torch.load(filename, map_location=map_location))
        print('load model!')

    def save_model(self, filename):
        directory = os.path.dirname(filename)
        os.makedirs(directory, exist_ok=True)
        torch.save(self.state_dict(), filename)

    def forward(self, obs, state=None, info={}):
        logits,_ = self.net(obs)
        logits = F.softmax(logits, dim=-1)
        return logits, state

class Critic(nn.Module):
    def __init__(self, is_gpu=is_gpu_default):
        super().__init__()
        self.is_gpu = is_gpu
        self.net = sdn_net(mode='critic')

    def load_model(self, filename):
        map_location = lambda storage, loc: storage
        self.load_state_dict(torch.load(filename, map_location=map_location))
        print('load model!')

    def save_model(self, filename):
        directory = os.path.dirname(filename)
        os.makedirs(directory, exist_ok=True)
        torch.save(self.state_dict(), filename)

    def forward(self, obs, state=None, info={}):
        v,_ = self.net(obs)
        return v

# Load các mô hình đã huấn luyện từ w00 đến w100
trained_models = {}

for wi in range(100, -1, -1):
    actor = Actor(is_gpu=is_gpu_default)
    critic = Critic(is_gpu=is_gpu_default)
    
    actor_file_path = f'save/pth-e{edge_num}/cloud{cloud_num}/{expn}/w{wi:03d}/ep{epoch:02d}-actor.pth'
    critic_file_path = f'save/pth-e{edge_num}/cloud{cloud_num}/{expn}/w{wi:03d}/ep{epoch:02d}-critic.pth'
    
    if os.path.exists(actor_file_path) and os.path.exists(critic_file_path):
        actor.load_model(actor_file_path)
        critic.load_model(critic_file_path)
        trained_models[wi] = (actor, critic)
    else:
        # Thực hiện xử lý nếu file không tồn tại, ví dụ: thông báo hoặc pass
        pass
# Tạo môi trường không huấn luyện với edge_num và cloud_num mong muốn
untrained_env = SDN_Env(conf_name=config, w=0.5, fc=4e9, fe=2e9, edge_num=edge_num, cloud_num=cloud_num)

# List to store solutions from all episodes for each wi
train_wi_delay_solutions = []
train_wi_link_utilisation_solutions = []
untrained_wi_delay_solutions = []
untrained_wi_link_utilisation_solutions = []

# Collect solutions from all episodes for each wi
for wi, (actor, critic) in trained_models.items():
    env = SDN_Env(conf_name=config, w=wi / 100.0, fc=4e9, fe=2e9, edge_num=edge_num, cloud_num=cloud_num)
    train_delay_solutions = []
    train_link_utilisation_solutions = []
    untrained_delay_solutions = []
    untrained_link_utilisation_solutions = []
    for _ in range(1):  # Number of episodes
        # For trained models
        state = env.reset()
        done = False
        trained_episode_solutions = []
        while not done:
            # Choose action using the actor model
            action, _ = actor(torch.tensor(state).float())
            action = action.argmax().item()
            next_state, reward, done, _ = env.step(action)
            # Collect objective values (delay, link utilization) for each step in the episode
            state = next_state
        delay, link_utilisation = env.estimate_performance()
        train_delay_solutions.append(delay)
        train_link_utilisation_solutions.append(link_utilisation)
        # For untrained models
        state = untrained_env.reset()
        done = False
        untrained_actor = Actor(is_gpu=is_gpu_default)
        untrained_critic = Critic(is_gpu=is_gpu_default)
        untrained_episode_solutions = []
        while not done:
            # Choose action using the actor model of the untrained model
            action, _ = untrained_actor(torch.tensor(state).float())
            action = action.argmax().item()
            next_state, reward, done, _ = untrained_env.step(action)
            state = next_state
        delay, link_utilisation = untrained_env.estimate_performance()
        untrained_delay_solutions.append(delay)
        untrained_link_utilisation_solutions.append(link_utilisation)    
    train_wi_delay_solutions.append(np.mean(train_delay_solutions, axis=0))
    train_wi_link_utilisation_solutions.append(np.mean(train_link_utilisation_solutions, axis=0))
    untrained_wi_delay_solutions.append(np.mean(untrained_delay_solutions, axis=0))
    untrained_wi_link_utilisation_solutions.append(np.mean(untrained_link_utilisation_solutions, axis=0))

train_solutions = pd.DataFrame(
    {
        "delay": train_wi_delay_solutions,
        "link_utilisation": train_wi_link_utilisation_solutions,
    }
)
untrained_solutions = pd.DataFrame(
    {
        "delay": untrained_wi_delay_solutions,
        "link_utilisation": untrained_wi_link_utilisation_solutions,
    }
)
# Compute the Pareto front for all solutions
trained_all_mask = paretoset(train_solutions, sense=["min", "min"])
untrained_all_mask = paretoset(untrained_solutions, sense=["min", "min"])
# Filter the list of solutions, keeping only the non-dominated solutions
trained_efficient_solutions = train_solutions[trained_all_mask]
untrained_efficient_solutions = untrained_solutions[untrained_all_mask]
trained_pareto_delay = []
trained_pareto_link_utilisation = []
untrained_pareto_delay = []
untrained_pareto_link_utilisation = []
# Extract delay and link utilization for trained efficient solutions
for index, row in trained_efficient_solutions.iterrows():
    trained_pareto_delay.append(row['delay'])
    trained_pareto_link_utilisation.append(row['link_utilisation'])
for index, row in untrained_efficient_solutions.iterrows():
    untrained_pareto_delay.append(row['delay'])
    untrained_pareto_link_utilisation.append(row['link_utilisation'])
# Scatter plot comparison
plt.figure(figsize=(8, 6))
plt.scatter(trained_pareto_delay, trained_pareto_link_utilisation, color='blue', label='Trained')
plt.scatter(untrained_pareto_delay, untrained_pareto_link_utilisation, color='red', label='Untrained')
plt.xlabel('Delay (s)')
plt.ylabel('Link Utilization (Mbps)')
plt.title('Comparison: Trained vs Untrained Pareto Front Models')
plt.grid(True)
plt.legend()
plt.show()