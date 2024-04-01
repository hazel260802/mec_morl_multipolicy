import numpy as np

# Define the softmax function
def softmax(x, tau):
    exp_x = np.exp(x / tau)
    return exp_x / np.sum(exp_x)

def run_softmax(env, edge_observation_space, cloud_observation_space, num_episodes, alpha=0.1, gamma=0.99, tau=0.1):
    """
    Run the softmax policy gradient algorithm to learn a policy for the given environment.

    Args:
    - env: The environment to run the algorithm on.
    - edge_observation_space: The observation space of the edge servers.
    - cloud_observation_space: The observation space of the cloud servers.
    - num_episodes: The number of episodes to run the algorithm for.
    - alpha: The learning rate (default is 0.1).
    - gamma: The discount factor (default is 0.99).
    - tau: The temperature parameter for the softmax function (default is 0.1).

    Returns:
    - avg_delays: A list containing the average delay per task for each episode.
    - avg_link_utilisations: A list containing the average link utilisation per task for each episode.
    """
    num_edge_states = edge_observation_space.shape[0]  # Number of edge states
    num_cloud_states = cloud_observation_space.shape[0]  # Number of cloud states
    num_edge_actions = edge_observation_space.shape[1]  # Number of edge actions
    num_cloud_actions = cloud_observation_space.shape[1]  # Number of cloud actions
    
    Q_edge = np.zeros((num_edge_states, num_edge_actions))  # Initialize Q-table for edge
    Q_cloud = np.zeros((num_cloud_states, num_cloud_actions))  # Initialize Q-table for cloud

    avg_delays = []  # List to store average delay per task for each episode
    avg_link_utilisations = []  # List to store average link utilisation per task for each episode
    
    for _ in range(num_episodes):
        state = env.reset()  # Reset the environment to initial state
        edge_state, cloud_state = state['edge_servers'], state['cloud_servers']  # Get the initial state for edge and cloud
        done = False  # Initialize flag indicating whether the episode has ended
 
        # Run the episode until termination
        while not done:
            # Calculate the mean values of each attribute across all servers for edge and cloud
            mean_edge_state = np.mean(edge_state, axis=0)
            mean_cloud_state = np.mean(cloud_state, axis=0)
            
            # Calculate the final probabilities using the softmax function
            edge_probs = softmax(mean_edge_state, tau)
            cloud_probs = softmax(mean_cloud_state, tau)
            
            # Sample actions from the softmax distributions
            edge_action = np.random.choice(num_edge_actions, p=edge_probs)
            cloud_action = np.random.choice(num_cloud_actions, p=cloud_probs)
            
            actions = np.array([edge_action, cloud_action])
            # Take a step in the environment based on the selected actions
            next_state, reward, done, _ = env.step(actions)
            
            # Update the action-value function Q for edge and cloud using the softmax policy gradient update
            if edge_state.shape[0] < num_edge_states and edge_action < num_edge_actions:
                Q_edge[edge_state.astype(int), edge_action] += alpha * (reward + gamma * np.max(Q_edge[next_state['edge_servers'].astype(int)]) - Q_edge[edge_state.astype(int), edge_action])
            
            if cloud_state.shape[0] < num_cloud_states and cloud_action < num_cloud_actions:
                Q_cloud[cloud_state.astype(int), cloud_action] += alpha * (reward + gamma * np.max(Q_cloud[next_state['cloud_servers'].astype(int)]) - Q_cloud[cloud_state.astype(int), cloud_action])
            
            # Transition to the next state
            edge_state, cloud_state = next_state['edge_servers'], next_state['cloud_servers']
        
        # Calculate average delay and link utilisation for this episode
        avg_delay, avg_link_utilisation = env.estimate_performance()
        
        avg_delays.append(avg_delay)  # Append the average delay per task for this episode to the list
        avg_link_utilisations.append(avg_link_utilisation)  # Append the average link utilisation per task for this episode to the list
    
    print("Average delay per task for each episode:", avg_delays)
    print("Average link utilisation per task for each episode:", avg_link_utilisations)
    return avg_delays, avg_link_utilisations
