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
from env import SDN_Env
from network import conv_mlp_net
from tianshou.utils.net.discrete import Actor, Critic
import matplotlib.pyplot as plt

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
        state = torch.tensor(obs).float()
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

import numpy as np

# Định nghĩa hàm tính Pareto dominance
def pareto_dominance(y1, y2):
    """
    Kiểm tra xem một điểm y1 có Pareto dominate điểm y2 hay không.
    """
    return all(y1_i <= y2_i for y1_i, y2_i in zip(y1, y2)) and any(y1_i < y2_i for y1_i, y2_i in zip(y1, y2))

# Load các mô hình đã huấn luyện từ w00 đến w100
trained_models = {}
for wi in range(100, 101, 2):
    actor = Actor(is_gpu=is_gpu_default)
    critic = Critic(is_gpu=is_gpu_default)
    actor.load_model(f'save/pth-e{edge_num}/cloud{cloud_num}/{expn}/w{wi:03d}/ep{epoch:02d}-actor.pth')
    critic.load_model(f'save/pth-e{edge_num}/cloud{cloud_num}/{expn}/w{wi:03d}/ep{epoch:02d}-critic.pth')
    trained_models[wi] = (actor, critic)

# Tính toán hiệu suất của các mô hình
performance = {}
for wi, (actor, critic) in trained_models.items():
    env = SDN_Env(conf_name=config, w=wi / 100.0, fc=4e9, fe=2e9, edge_num=edge_num, cloud_num=cloud_num)
    avg_link_utilisation = []
    avg_delay = []
    for _ in range(100):  # Số lượng thử nghiệm
        state = env.reset()
        done = False
        while not done:
            action = actor(torch.tensor(state).float())[0].argmax().item()
            next_state, reward, done, _ = env.step(action)
            state = next_state
        delay, link_utilisation = env.estimate_performance()
        avg_link_utilisation.append(link_utilisation)
        avg_delay.append(delay)
    # Tính toán giá trị trung bình của delay và link_utilisation
    avg_delay = np.mean(avg_delay)
    avg_link_utilisation = np.mean(avg_link_utilisation)
    # Lưu thông số hiệu suất vào performance dictionary
    performance[wi] = (avg_delay, avg_link_utilisation)

# Tách các điểm dữ liệu từ dictionary performance
pareto_delay = []
pareto_link_utilisation = []
for wi, (delay, link_utilisation) in performance.items():
    pareto_delay.append(delay)
    pareto_link_utilisation.append(link_utilisation)

# Vẽ biểu đồ scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(pareto_delay, pareto_link_utilisation, color='red', label='Pareto Front')
plt.xlabel('Delay')
plt.ylabel('Link Utilization')
plt.title('Pareto Front Models: Delay vs Link Utilization')
plt.grid(True)
plt.legend()
plt.show()