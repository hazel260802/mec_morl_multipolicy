import torch
import json
from env import SDN_Env  # Import your environment class
from network import conv_mlp_net  # Import your network architecture
from train import edge_num, cloud_num, expn  # Import your training parameters
# Define the function to load the trained model
def load_model(model, filename):
    # Load the model state dict from the file
    model.load_state_dict(torch.load(filename))
    print('Model loaded successfully.')

# Define the function to simulate the network using an actor model
def simulate_network(actor, env, num_episodes=10):
    delay_data = []
    link_utilization_data = []

    for _ in range(num_episodes):
        obs = env.reset()
        total_delay = 0
        total_link_utilization = 0

        done = False
        while not done:
            # Select an action using the actor model
            action, _ = actor(torch.tensor(obs).float())
            action = action.argmax().item()

            # Simulate the network using the selected action
            next_obs, reward, done, info = env.step(action)

            # Update delay and link utilization data
            total_delay += info['delay']
            total_link_utilization += info['link_utilization']

            obs = next_obs

        # Calculate average delay and link utilization for the episode
        avg_delay, avg_link_utilization = env.estimate_performance()

        # Append to the data lists
        delay_data.append(avg_delay)
        link_utilization_data.append(avg_link_utilization)

    return delay_data, link_utilization_data

# Define the path to the directory containing actor models
actor_models_dir = 'save/pth-e%d/cloud%d/%s/w100/' % (edge_num, cloud_num, expn)

# Create an empty list to store actor models
actor_models = []

# Load each actor model from ep00 to ep10 and append to the list
for i in range(11):
    # Build the path to the actor model file
    actor_model_path = actor_models_dir + 'ep%02d-actor.pth' % i
    
    # Create a new actor model
    actor_model = conv_mlp_net()  # Modify this line according to your actor network architecture
    
    # Load the actor model
    load_model(actor_model, actor_model_path)
    
    # Append the loaded actor model to the list
    actor_models.append(actor_model)

# Create an instance of your environment
env = SDN_Env(conf_name='multi-edge', w=0.5, fc=4e9, fe=2e9, edge_num=1, cloud_num=1)  # Modify parameters accordingly

# Simulate the network using each loaded actor model and save the results
num_episodes = 10  # Modify as needed

for idx, actor_model in enumerate(actor_models):
    delay_data, link_utilization_data = simulate_network(actor_model, env, num_episodes)
    
    # Save the delay and link utilization data to JSON files
    with open('delay_data_ep%02d.json' % idx, 'w') as f:
        json.dump(delay_data, f)

    with open('link_utilization_data_ep%02d.json' % idx, 'w') as f:
        json.dump(link_utilization_data, f)