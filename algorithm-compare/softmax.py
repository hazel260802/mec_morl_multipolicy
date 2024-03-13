import gym
import numpy as np

# Tạo môi trường
env = gym.make('SDN_Env')

# Thiết lập siêu tham số
alpha = 0.1  # Tỷ lệ học
gamma = 0.99  # Hệ số giảm giá
temperature = 0.1  # Nhiệt độ trong softmax
num_episodes = 1000  # Số lượng tập huấn luyện

# Khởi tạo ma trận Q với giá trị ban đầu là 0
num_states = env.observation_space.n
num_actions = env.action_space.n
Q = np.zeros((num_states, num_actions))

# Huấn luyện mô hình Q-learning
for _ in range(num_episodes):
    state = env.reset()
    done = False
    while not done:
        # Tính xác suất của mỗi hành động bằng softmax
        logits = Q[state] / temperature
        probabilities = np.exp(logits) / np.sum(np.exp(logits))
        
        # Chọn hành động theo phân phối xác suất
        action = np.random.choice(num_actions, p=probabilities)
        
        # Thực hiện hành động và nhận lại các thông tin từ môi trường
        next_state, reward, done, _ = env.step(action)

        # Cập nhật giá trị của ma trận Q
        Q[state, action] += alpha * (reward + gamma * np.max(Q[next_state]) - Q[state, action])

        # Cập nhật trạng thái hiện tại
        state = next_state

# Kiểm thử mô hình
total_rewards = []
for _ in range(100):
    state = env.reset()
    total_reward = 0
    done = False
    while not done:
        # Chọn hành động có giá trị Q cao nhất
        action = np.argmax(Q[state])
        
        # Thực hiện hành động và nhận lại các thông tin từ môi trường
        next_state, reward, done, _ = env.step(action)
        total_reward += reward
        state = next_state
    total_rewards.append(total_reward)

print("Average total reward:", np.mean(total_rewards))
