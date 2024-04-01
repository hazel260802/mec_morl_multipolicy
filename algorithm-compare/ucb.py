import numpy as np

def ucb1(Q, N, t, c=2):
    """
    UCB1 algorithm for action selection.
    
    Args:
    - Q: Array of action values.
    - N: Array of visit counts for each action.
    - t: Total number of time steps so far.
    - c: Exploration-exploitation trade-off parameter.
    
    Returns:
    - Action selected by UCB1 algorithm.
    """
    # Calculate UCB values for each action
    ucb_values = Q + c * np.sqrt(np.log(t) / (N + 1e-8))  # Adding a small value to prevent division by zero
    return np.argmax(ucb_values)  # Select action with the highest UCB value

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

def run_ucb1(env, edge_observation_space, cloud_observation_space, num_episodes, alpha=0.1, gamma=0.99, c=2):
    """
    Runs the UCB1 algorithm for action selection in a given environment.

    Args:
    - env: The environment.
    - edge_observation_space: The observation space of the edge servers.
    - cloud_observation_space: The observation space of the cloud servers.
    - num_episodes: Number of episodes to run the algorithm.
    - alpha: Learning rate.
    - gamma: Discount factor for future rewards.
    - c: Exploration-exploitation trade-off parameter for UCB1.

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

    N_edge = np.zeros((num_edge_states, num_edge_actions))  # Count of visits for each action
    N_cloud = np.zeros((num_cloud_states, num_cloud_actions))  # Count of visits for each action
    
    avg_delays = []  # List to store average delay per task for each episode
    avg_link_utilisations = []  # List to store average link utilisation per task for each episode
    
    for _ in range(num_episodes):
        state = env.reset()  # Reset the environment to initial state
        edge_state, cloud_state = state['edge_servers'], state['cloud_servers']  # Get the initial state for edge and cloud
        done = False  # Initialize flag indicating whether the episode has ended
        t = 0  # Total time steps
        while not done:
            t += 1  # Increment time step
            
            # Preprocess the current state
            processed_edge_state = preprocess_state(edge_state)
            processed_cloud_state = preprocess_state(cloud_state)
            
            # Action selection for edge
            if np.all((0 <= processed_edge_state) & (processed_edge_state < num_edge_states)):
                edge_action = ucb1(Q_edge[processed_edge_state.astype(int)], N_edge[processed_edge_state.astype(int)], t, c)
            else:
                edge_action = np.random.choice(num_edge_actions)  # Random action if state is out of bounds
            
            # Action selection for cloud
            if np.all((0 <= processed_cloud_state) & (processed_cloud_state < num_cloud_states)):
                cloud_action = ucb1(Q_cloud[processed_cloud_state.astype(int)], N_cloud[processed_cloud_state.astype(int)], t, c)
            else:
                cloud_action = np.random.choice(num_cloud_actions)  # Random action if state is out of bounds
            
            actions = np.array([edge_action, cloud_action])
            # Take a step in the environment based on the selected actions
            next_state, reward, done, _ = env.step(actions)
            
            # Update action-value estimates and visit counts for edge using Q-learning update rule
            if edge_state.shape[0] in range(num_edge_states) and edge_action in range(num_edge_actions):
                N_edge[processed_edge_state.astype(int), edge_action] += 1
                Q_edge[processed_edge_state.astype(int), edge_action] += alpha * (reward + gamma * np.max(Q_edge[next_state['edge_servers'].astype(int)]) - Q_edge[processed_edge_state.astype(int), edge_action])
            
            # Update action-value estimates and visit counts for cloud using Q-learning update rule
            if cloud_state.shape[0] in range(num_cloud_states) and cloud_action in range(num_cloud_actions):
                N_cloud[processed_cloud_state.astype(int), cloud_action] += 1
                Q_cloud[processed_cloud_state.astype(int), cloud_action] += alpha * (reward + gamma * np.max(Q_cloud[next_state['cloud_servers'].astype(int)]) - Q_cloud[processed_cloud_state.astype(int), cloud_action])
            
            # Transition to the next state
            edge_state, cloud_state = next_state['edge_servers'], next_state['cloud_servers']
        
        # Calculate average delay and link utilisation for this episode
        avg_delay, avg_link_utilisation = env.estimate_performance()
        
        avg_delays.append(avg_delay)  # Append the average delay per task for this episode to the list
        avg_link_utilisations.append(avg_link_utilisation)  # Append the average link utilisation per task for this episode to the list
    
    print("Average delay per task for each episode:", avg_delays)
    print("Average link utilisation per task for each episode:", avg_link_utilisations)
    return avg_delays, avg_link_utilisations
