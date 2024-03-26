from copy import deepcopy
from env import SDN_Env
from tianshou.env import DummyVectorEnv
from train import config, edge_num, cloud_num, test_num, actor, critic
import json
# Khởi tạo mô hình chưa được huấn luyện
untrained_actor = deepcopy(actor)
untrained_critic = deepcopy(critic)

class UntrainedActorCritic(ts.utils.net.common.ActorCritic):
    def forward(self, obs, state=None, info={}):
        logits, value = self.actor(obs), self.critic(obs)
        return logits, value

untrained_actor_critic = UntrainedActorCritic(untrained_actor, untrained_critic)

# Khởi tạo môi trường và thu thập dữ liệu cho mô hình chưa được huấn luyện
untrained_test_envs = DummyVectorEnv(
    [lambda: SDN_Env(conf_name=config, w=0.0, fc=4e9, fe=2e9, edge_num=edge_num, cloud_num=cloud_num) for _ in range(test_num)]
)
untrained_collector = ts.data.Collector(
    policy=untrained_actor_critic,
    env=untrained_test_envs,
    buffer=None  # Không sử dụng buffer trong quá trình đánh giá
)

# Collect data và đánh giá hiệu suất của mô hình chưa được huấn luyện
untrained_ave_delay_per_episode = []
untrained_ave_link_util_per_episode = []

for _ in range(test_num):
    # Thực hiện hành động bằng cách sử dụng chính sách chưa được huấn luyện và thu thập dữ liệu
    untrained_collector.collect(n_episode=1)
    # Tính toán trung bình độ trễ và trung bình sử dụng băng thông của liên kết cho mỗi tập tin JSON
    for env in untrained_test_envs.venv:
        # Do something with each environment 'env'
        untrained_ave_delay, untrained_ave_link_util = env.estimate_performance()
    untrained_ave_delay_per_episode.append(untrained_ave_delay)
    untrained_ave_link_util_per_episode.append(untrained_ave_link_util)

# Lưu trung bình độ trễ và trung bình sử dụng băng thông của liên kết cho mỗi episode vào tệp tin JSON cho mô hình chưa được huấn luyện
with open('result/ave_delay_per_episode_untrained.json', 'w') as f:
    json.dump(untrained_ave_delay_per_episode, f)

with open('result/ave_link_util_per_episode_untrained.json', 'w') as f:
    json.dump(untrained_ave_link_util_per_episode, f)
