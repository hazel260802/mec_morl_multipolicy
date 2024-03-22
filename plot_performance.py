import matplotlib.pyplot as plt
import json

# Load data from JSON files for both trained and untrained models
with open('result/ave_delay_per_episode.json', 'r') as f:
    ave_delay_trained = json.load(f)

with open('result/ave_link_util_per_episode.json', 'r') as f:
    ave_link_util_trained = json.load(f)

# Load data for untrained model (if available)
# You need to provide the JSON files for the untrained model
with open('result/ave_delay_per_episode_untrained.json', 'r') as f:
    ave_delay_untrained = json.load(f)

with open('result/ave_link_util_per_episode_untrained.json', 'r') as f:
    ave_link_util_untrained = json.load(f)

# Plot average delay
plt.plot(ave_delay_trained, label='Trained Model')
if ave_delay_untrained:
    plt.plot(ave_delay_untrained, label='Untrained Model', linestyle='--')
plt.xlabel('Episode')
plt.ylabel('Average Delay')
plt.title('Average Delay Over Episodes')
plt.legend()
plt.show()

# Plot average link utilization
plt.plot(ave_link_util_trained, label='Trained Model')
if ave_link_util_untrained:
    plt.plot(ave_link_util_untrained, label='Untrained Model', linestyle='--')
plt.xlabel('Episode')
plt.ylabel('Average Link Utilization')
plt.title('Average Link Utilization Over Episodes')
plt.legend()
plt.show()
