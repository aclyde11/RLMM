import numpy as np
# Initialize q-table values to 0
Q = np.zeros((state_size, action_size))

import random
# Set the percent you want to explore
epsilon = 0.2
if random.uniform(0, 1) < epsilon:
    """
    Explore: select a random action
    """
else:
    """
    Exploit: select the action with max value (future reward)
    """

    # Update q values
Q[state, action] = Q[state, action] + lr * (reward + gamma * np.max(Q[new_state, :]) — Q[state, action])