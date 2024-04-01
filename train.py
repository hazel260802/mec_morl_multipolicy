import gym as gym
from tianshou.data import Batch
from typing import Optional, Union, Any
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
from tianshou.env.gym_wrappers import MultiDiscreteToDiscrete
from tianshou.env import VectorEnvWrapper

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
            self.edge_net = conv_mlp_net(conv_in=INPUT_CH, conv_ch=FEATURE_CH, mlp_in=edge_num * FEATURE_CH,\
                                    mlp_ch=MLP_CH, out_ch=edge_num, block_num=3)
            self.cloud_net = conv_mlp_net(conv_in=INPUT_CH, conv_ch=FEATURE_CH, mlp_in=cloud_num * FEATURE_CH,\
                                    mlp_ch=MLP_CH, out_ch=cloud_num, block_num=3)
        else:
            self.network = conv_mlp_net(conv_in=INPUT_CH, conv_ch=FEATURE_CH, mlp_in=(edge_num+cloud_num)*FEATURE_CH,\
                                    mlp_ch=MLP_CH, out_ch=edge_num+cloud_num, block_num=3)

    def load_model(self, filename):
        map_location = lambda storage, loc: storage
        self.load_state_dict(torch.load(filename, map_location=map_location))
        print('load model!')

    def save_model(self, filename):
        directory = os.path.dirname(filename)
        os.makedirs(directory, exist_ok=True)
        torch.save(self.state_dict(), filename)

    def forward(self, obs, state=None, info={}):
        state = obs
        state = torch.tensor(state).float()
        if self.is_gpu:
            state = state.cuda()
        # Chỉ trả về kết quả từ mạng tương ứng với chế độ
        logits = None
        if self.mode == 'actor':
            # Kiểm tra xem self.edge_net có được khởi tạo hay không
            if hasattr(self, 'edge_net') and self.edge_net is not None:
                logits = self.edge_net(state)
            # Kiểm tra xem self.cloud_net có được khởi tạo hay không
            elif hasattr(self, 'cloud_net') and self.cloud_net is not None:
                logits = self.cloud_net(state)
        else:
            logits = self.network(state)
        # print("State: ", state)
        # print("Logits: ", logits)
        return logits, state


class Actor(nn.Module):
    def __init__(self, is_gpu=is_gpu_default, dist_fn=None):
        super().__init__()
        self.is_gpu = is_gpu
        self.edge_net = sdn_net(mode='actor')
        self.cloud_net = sdn_net(mode='actor')
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
        logits_edge, _ = self.edge_net(obs['edge_servers'])
        print(logits_edge)
        logits_edge = F.softmax(logits_edge, dim=-1)

        logits_cloud, _ = self.cloud_net(obs['cloud_servers'])
        logits_cloud = F.softmax(logits_cloud, dim=-1)
        # print(logits_edge, logits_cloud)
        return logits_edge, logits_cloud

class Critic(nn.Module):
    def __init__(self, is_gpu=is_gpu_default):
        # print(f"Batch keys: {self.__dict__.keys()}")
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
class PPOPolicy(ts.policy.PPOPolicy):
    def __init__(self, actor, critic, optim, dist_fn, discount_factor=0.99, gae_lambda=0.95,
                 max_grad_norm=None, eps_clip=0.2, vf_coef=0.5, ent_coef=0.0,
                 reward_normalization=False, advantage_normalization=False,
                 action_scaling=True, recompute_advantage=False, dual_clip=None,
                 value_clip=False, action_space=None, lr_scheduler=None,
                 update_per_collect=1):
        super().__init__(actor, critic, optim, dist_fn)

    def forward(
        self,
        batch: Batch,
        state: Optional[Union[dict, Batch, np.ndarray]] = None,
        **kwargs: Any,
    ) -> Batch:
        logits_edge, logits_cloud = self.actor(batch.obs, state=state)
        dist_edge, dist_cloud = self.dist_fn(logits_edge, logits_cloud)  
        print(dist_edge, dist_cloud)
        if self._deterministic_eval and not self.training:
            act_edge = logits_edge.argmax(-1)
            act_cloud = logits_cloud.argmax(-1) 
        else:
            act_edge = dist_edge.sample()
            act_cloud = dist_cloud.sample()
        print(act_edge, act_cloud)
        # Ensure actions are within the valid range of action space
        act_edge = torch.clamp(act_edge, 0, edge_num - 1)
        act_cloud = torch.clamp(act_cloud, 0, cloud_num - 1)

        # Combine actions into a single tensor
        action = torch.stack([act_edge, act_cloud], dim=-1)
        return Batch(
            logits=(logits_edge, logits_cloud),
            act=action,
            state=(None, None),
            dist=(dist_edge, dist_cloud)
        )



actor = Actor(is_gpu=is_gpu_default)
critic = Critic(is_gpu=is_gpu_default)
actor_critic = ts.utils.net.common.ActorCritic(actor, critic)
optim = torch.optim.Adam(actor_critic.parameters(), lr=lr)

dist = torch.distributions.Categorical

action_space = gym.spaces.MultiDiscrete([edge_num, cloud_num])
if lr_decay:
    lr_scheduler = LambdaLR(
        optim, lr_lambda=lambda epoch: lr_decay ** (epoch - 1)
    )
else:
    lr_scheduler = None

def custom_dist_fn(logits_edge, logits_cloud):
    return (
        dist(logits=logits_edge),
        dist(logits=logits_cloud)
    )

policy = PPOPolicy(actor, critic, optim, custom_dist_fn,
                   discount_factor=gamma, max_grad_norm=max_grad_norm,
                   eps_clip=eps_clip, vf_coef=vf_coef,
                   ent_coef=ent_coef, reward_normalization=rew_norm,
                   advantage_normalization=norm_adv, recompute_advantage=recompute_adv,
                   dual_clip=dual_clip, value_clip=value_clip,
                   gae_lambda=gae_lambda, action_space=action_space,
                   lr_scheduler=lr_scheduler,
                  )

for i in range(101):
    try:
        os.mkdir('save/pth-e%d/' % (edge_num) + 'cloud%d/' % (cloud_num) + expn + '/w%03d' % (i))
    except:
        pass


for wi in range(100, 0 - 1, -2):

    if wi == 100:
        epoch_a = epoch * 10
    else:
        epoch_a = epoch

    # train_envs = DummyVectorEnv(
    #     [lambda: MultiDiscreteToDiscrete(SDN_Env(conf_name=config, w=wi / 100.0, fc=4e9, fe=2e9, edge_num=edge_num, cloud_num=cloud_num)) for _ in range(train_num)])
    # test_envs = DummyVectorEnv(
    #     [lambda: MultiDiscreteToDiscrete(SDN_Env(conf_name=config, w=wi / 100.0, fc=4e9, fe=2e9, edge_num=edge_num, cloud_num=cloud_num)) for _ in range(test_num)])
    train_envs = DummyVectorEnv(
        [lambda: SDN_Env(conf_name=config, w=wi / 100.0, fc=4e9, fe=2e9, edge_num=edge_num, cloud_num=cloud_num) for _ in range(train_num)])
    test_envs = DummyVectorEnv(
        [lambda: SDN_Env(conf_name=config, w=wi / 100.0, fc=4e9, fe=2e9, edge_num=edge_num, cloud_num=cloud_num) for _ in range(test_num)])
    buffer = ts.data.VectorReplayBuffer(buffer_size, train_num)
    train_collector = ts.data.Collector(
        policy=policy,
        env=train_envs,
        buffer=buffer,
    )
    # print(train_collector.env.action_space)
    test_collector = ts.data.Collector(policy, test_envs)
    train_collector.collect(n_episode=train_num)

    def save_best_fn(policy, epoch, env_step, gradient_step):
        pass

    def test_fn(epoch, env_step):
        policy.actor.save_model('save/pth-e%d/' % (edge_num) + 'cloud%d/' % (cloud_num) + expn + '/w%03d/ep%02d-actor.pth' % (wi, epoch))
        policy.critic.save_model('save/pth-e%d/' % (edge_num) + 'cloud%d/' % (cloud_num) + expn + '/w%03d/ep%02d-critic.pth' % (wi, epoch))

    def train_fn(epoch, env_step):
        pass

    def reward_metric(rews):
        return rews

    result = ts.trainer.onpolicy_trainer(
        policy=policy,
        train_collector=train_collector,
        test_collector=test_collector,
        max_epoch=epoch_a,
        step_per_epoch=step_per_epoch,
        repeat_per_collect=repeat_per_collect,
        episode_per_test=test_num,
        batch_size=batch_size,
        step_per_collect=None,
        episode_per_collect=episode_per_collect,
        train_fn=train_fn,
        test_fn=test_fn,
        save_best_fn=save_best_fn(policy, epoch, 0, 0),
        stop_fn=None,
        save_checkpoint_fn=save_best_fn(policy, epoch, 0, 0),
        reward_metric=reward_metric,
        logger=logger,
    )
        
 