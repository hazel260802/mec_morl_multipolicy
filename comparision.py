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

cloud_num = 8
edge_num = 8
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

for wi in range(0, 101, 2):
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
            # Choose action using the actor model
            action, _ = actor(torch.tensor(state).float())
            action = action.argmax().item()
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

# Tạo môi trường không huấn luyện với edge_num và cloud_num mong muốn
untrained_env = SDN_Env(conf_name=config, w=0.5, fc=4e9, fe=2e9, edge_num=edge_num, cloud_num=cloud_num)

# Tính toán hiệu suất của các mô hình đã huấn luyện trước đó trên môi trường không huấn luyện
trained_performance = {}
for wi, (actor, critic) in trained_models.items():
    avg_link_utilisation = []
    avg_delay = []
    for _ in range(100):  # Số lượng thử nghiệm
        state = untrained_env.reset()
        done = False
        while not done:
            # Choose action using the actor model
            action, _ = actor(torch.tensor(state).float())
            action = action.argmax().item()
            next_state, reward, done, _ = untrained_env.step(action)
            state = next_state
        delay, link_utilisation = untrained_env.estimate_performance()
        avg_link_utilisation.append(link_utilisation)
        avg_delay.append(delay)
    # Tính toán giá trị trung bình của delay và link_utilisation
    avg_delay = np.mean(avg_delay)
    avg_link_utilisation = np.mean(avg_link_utilisation)
    # Lưu thông số hiệu suất vào trained_performance dictionary
    trained_performance[wi] = (avg_delay, avg_link_utilisation)

# Tách các điểm dữ liệu từ dictionary trained_performance
trained_pareto_delay = []
trained_pareto_link_utilisation = []
for wi, (delay, link_utilisation) in trained_performance.items():
    trained_pareto_delay.append(delay)
    trained_pareto_link_utilisation.append(link_utilisation)

# Vẽ biểu đồ scatter plot so sánh giữa các mô hình đã huấn luyện và môi trường không huấn luyện
plt.figure(figsize=(8, 6))
plt.scatter(pareto_delay, pareto_link_utilisation, color='red', label='Untrained')
plt.scatter(trained_pareto_delay, trained_pareto_link_utilisation, color='blue', label='Trained')
plt.xlabel('Delay')
plt.ylabel('Link Utilization')
plt.title('Comparison: Trained vs Untrained Pareto Front Models')
plt.grid(True)
plt.legend()
plt.show()

# Vẽ biểu đồ đường riêng cho trained preferences
plt.figure(figsize=(8, 6))
plt.plot(trained_pareto_delay, trained_pareto_link_utilisation, color='blue', label='MORL')
plt.xlabel('Delay')
plt.ylabel('Link uitilization ')
plt.title('MORL Model Pareto Front')
plt.grid(True)
plt.legend()
plt.show()
