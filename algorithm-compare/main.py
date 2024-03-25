import gym
import numpy as np
import matplotlib.pyplot as plt
from egreedy import run_egreedy
from softmax import run_softmax
from ucb import run_ucb1
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from env.SDN_env import SDN_Env

config = 'multi-edge'
cloud_num = 1
edge_num = 1

def main():
    # Initialize the SDN environment
    env = SDN_Env(conf_file='../env/config1.json', conf_name=config, w=1.0, fc=4e9, fe=2e9, edge_num=edge_num, cloud_num=cloud_num)
    # Set hyperparameters
    num_episodes = 1000  
    
    observation_space = env.get_obs()
    # Run ε-Greedy algorithm
    egreedy_delays, egreedy_link_utilisations = run_egreedy(env, observation_space, num_episodes=num_episodes)
    # Tính toán giá trị trung bình trên mỗi phút
    egreedy_delays_avg = [np.mean(egreedy_delays[i:i+200]) for i in range(0, len(egreedy_delays), 200)]
    egreedy_link_utilisations_avg = [np.mean(egreedy_link_utilisations[i:i+200]) for i in range(0, len(egreedy_link_utilisations), 200)]

    # Run Softmax algorithm
    softmax_delays, softmax_link_utilisations = run_softmax(env, observation_space, num_episodes=num_episodes)
    # Tính toán giá trị trung bình trên mỗi phút
    softmax_delays_avg = [np.mean(softmax_delays[i:i+200]) for i in range(0, len(softmax_delays), 200)]
    softmax_link_utilisations_avg = [np.mean(softmax_link_utilisations[i:i+200]) for i in range(0, len(softmax_link_utilisations), 200)]

    # Run UCB1 algorithm
    ucb_delays, ucb_link_utilisations = run_ucb1(env, observation_space, num_episodes=num_episodes)
    # Tính toán giá trị trung bình trên mỗi phút
    ucb_delays_avg = [np.mean(ucb_delays[i:i+200]) for i in range(0, len(ucb_delays), 200)]
    ucb_link_utilisations_avg = [np.mean(ucb_link_utilisations[i:i+200]) for i in range(0, len(ucb_link_utilisations), 200)]
    
    # # Run PPO algorithm
    # ave_delay_filename = '../result/ave_delay_per_episode.json'
    # ave_link_util_filename = '../result/ave_link_util_per_episode.json'

    # # Đọc dữ liệu từ các tệp tin JSON
    # ppo_delays = load_results_from_json(ave_delay_filename)
    # ppo_link_utilisations = load_results_from_json(ave_link_util_filename)
    # Plotting the results
    plt.figure(figsize=(10, 5))

    # Tạo mảng chứa giá trị của trục x tính theo phút
    egreedy_delays_avg = [i * 20 for i in range(len(egreedy_delays_avg))]
    softmax_delays_avg = [i * 20 for i in range(len(softmax_delays_avg))]
    ucb_delays_avg = [i * 20 for i in range(len(ucb_delays_avg))]
    
    # Plot ε-Greedy
    plt.plot(egreedy_delays_avg, egreedy_link_utilisations_avg, label='ε-Greedy')

    # Plot Softmax
    plt.plot(softmax_delays_avg, softmax_link_utilisations_avg, label='Softmax')

    # Plot UCB1
    plt.plot(ucb_delays_avg, ucb_link_utilisations_avg, label='UCB1')

    # Plot PPO
    # plt.plot(ppo_delays, ppo_link_utilisations, label='PPO')
    plt.xlabel('Task Delay')
    plt.ylabel('Link Utilisation')
    plt.title('Comparison of Task Delay and Link Utilisation for ε-Greedy, Softmax, and UCB1 Algorithms')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
