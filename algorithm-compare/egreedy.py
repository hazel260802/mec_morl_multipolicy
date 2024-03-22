import numpy as np

def run_egreedy(env, observation_space, num_episodes=1000, alpha=0.1, gamma=0.99, epsilon=0.1):

    num_states = observation_space.shape[0]
    num_actions = env.action_space.n
    Q = np.zeros((num_states, num_actions))

    rewards = []
    for _ in range(num_episodes):
        state = env.reset()
        # print("State:", state)
        total_reward = 0
        done = False
        while not done:
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                # print("State shape:", state.shape)
                # print("Q shape:", Q.shape)
                if np.all((0 <= state) & (state < num_states)):
                    max_action = np.argmax(Q[state.astype(int)])
                    if max_action in range(num_actions):
                        action = max_action
                # print("Action(egreedy):", action)
            next_state, reward, done, _ = env.step(action)
            total_reward += reward
            if state.shape[0] in range(num_states) and action in range(num_actions):
                Q[state.astype(int), action] += alpha * (reward + gamma * np.max(Q[next_state.astype(int)]) - Q[state.astype(int), action])
            state = next_state
        rewards.append(total_reward)
    
    return rewards
