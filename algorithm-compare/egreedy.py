import numpy as np

def preprocess_state(state):
    """
    Preprocesses the current state of each server.

    Parameters:
    - state: The current state of each server.

    Returns:
    - processed_state: The preprocessed state of each server.
    """
    # Calculate the mean of each column (parameter) across all servers
    mean_state = np.mean(state, axis=0)
    return mean_state

def run_egreedy(env, edge_observation_space, cloud_observation_space, num_episodes=1000, alpha=0.1, gamma=0.99, epsilon=0.1):
    """
    Runs the epsilon-greedy algorithm for reinforcement learning.

    Parameters:
    - env: The environment.
    - edge_observation_space: The observation space of the edge servers.
    - cloud_observation_space: The observation space of the cloud servers.
    - num_episodes: Number of episodes to run the algorithm.
    - alpha: Learning rate.
    - gamma: Discount factor for future rewards.
    - epsilon: The probability of choosing a random action (exploration rate).

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
        
        while not done:
            if np.random.rand() < epsilon:  # With probability epsilon, choose a random action
                edge_action, cloud_action = env.action_space.sample()
            else:
                # Preprocess the current state for edge
                processed_edge_state = preprocess_state(edge_state)
                
                # Choose the action with the highest Q-value for the current edge state
                # If the state is out of bounds, choose a random action
                if np.all((0 <= processed_edge_state) & (processed_edge_state < num_edge_states)):
                    max_edge_action = np.argmax(Q_edge[processed_edge_state.astype(int)])
                    if max_edge_action in range(num_edge_actions):
                        edge_action = max_edge_action
                        
                # Preprocess the current state for cloud
                processed_cloud_state = preprocess_state(cloud_state)
                print(processed_edge_state, processed_cloud_state)
                # Choose the action with the highest Q-value for the current cloud state
                # If the state is out of bounds, choose a random action
                if np.all((0 <= processed_cloud_state) & (processed_cloud_state < num_cloud_states)):
                    max_cloud_action = np.argmax(Q_cloud[processed_cloud_state.astype(int)])
                    if max_cloud_action in range(num_cloud_actions):
                        cloud_action = max_cloud_action
            # print(edge_action, cloud_action)
            actions = np.array([edge_action, cloud_action])
            next_state, reward, done, _ = env.step(actions)  # Take a step in the environment

            if edge_state.shape[0] in range(num_edge_states) and edge_action in range(num_edge_actions):
                # Update Q-value using the Q-learning update rule for edge
                Q_edge[processed_edge_state.astype(int), edge_action] += alpha * (reward + gamma * np.max(Q_edge[next_state['edge_servers'].astype(int)]) - Q_edge[processed_edge_state.astype(int), edge_action])
            
            if cloud_state.shape[0] in range(num_cloud_states) and cloud_action in range(num_cloud_actions):
                # Update Q-value using the Q-learning update rule for cloud
                Q_cloud[processed_cloud_state.astype(int), cloud_action] += alpha * (reward + gamma * np.max(Q_cloud[next_state['cloud_servers'].astype(int)]) - Q_cloud[processed_cloud_state.astype(int), cloud_action])
            
            state = next_state # Move to the next state
        
        # Calculate average delay and link utilisation for this episode
        avg_delay, avg_link_utilisation = env.estimate_performance()
        
        avg_delays.append(avg_delay)  # Append the average delay per task for this episode to the list
        avg_link_utilisations.append(avg_link_utilisation)  # Append the average link utilisation per task for this episode to the list
        
    print("Average delay per task for each episode:", avg_delays)
    print("Average link utilisation per task for each episode:", avg_link_utilisations)
    return avg_delays, avg_link_utilisations
