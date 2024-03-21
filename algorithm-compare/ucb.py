import numpy as np
import math

def select_action_ucb(Q, N, state, t):
    num_actions = Q.shape[1]
    ucb_values = np.zeros(num_actions)
    for a in range(num_actions):
        if N[state, a] == 0:
            ucb_values[a] = float('inf')  # Assign infinity for unexplored actions
        else:
            q = Q[state, a]
            ucb_values[a] = q + 2 * math.log(t) / N[state, a]
    action = np.argmax(ucb_values)
    return min(action, num_actions - 1)  # Ensure action is within the valid range
