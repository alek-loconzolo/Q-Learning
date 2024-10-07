import numpy as np
from tqdm import tqdm
import gymnasium as gym
import matplotlib.pyplot as plt
import json
import matrix_mdp
import sys
import seaborn as sns

nS = 16
nA = 4

slip_prob = 0.1  

actions = ["up", "down", "left", "right"]  # Human readable labels for actions

p_0 = np.array([0 for _ in range(nS)], dtype=np.float64)
p_0[12] = 1

P = np.zeros((nS, nS, nA), dtype=np.float64)


def valid_neighbors(i, j):
    neighbors = {}
    if i > 0:
        neighbors[0] = (i - 1, j)
    if i < 3:
        neighbors[1] = (i + 1, j)
    if j > 0:
        neighbors[2] = (i, j - 1)
    if j < 3:
        neighbors[3] = (i, j + 1)
    return neighbors


for i in range(4):
    for j in range(4):
        if i == 0 and j == 2:
            continue  
        if i == 3 and j == 1:
            continue  

        neighbors = valid_neighbors(i, j)
        for a in range(nA):
            if a in neighbors:
                P[neighbors[a][0] * 4 + neighbors[a][1], i * 4 + j, a] = 1 - slip_prob
                for b in neighbors:
                    if b != a:
                        P[neighbors[b][0] * 4 + neighbors[b][1], i * 4 + j, a] = (
                            slip_prob / float(len(neighbors.items()) - 1)
                        )

#################################################################
# REWARD MATRIX

# In this implementation, you only get the reward if you *intended* to get to
# the target state with the corresponding action, but not through slipping.

#################################################################

R = np.zeros((nS, nS, nA), dtype=np.float64)

R[2, 1, 3] = 2000
R[2, 3, 2] = 2000
R[2, 6, 0] = 2000

R[13, 9, 1] = 2
R[13, 14, 2] = 2
R[13, 12, 3] = 2

R[11, 15, 0] = -100
R[11, 7, 1] = -100
R[11, 10, 3] = -100
R[10, 14, 0] = -100
R[10, 6, 1] = -100
R[10, 11, 2] = -100
R[10, 9, 3] = -100
R[9, 10, 2] = -100
R[9, 13, 0] = -100
R[9, 5, 1] = -100
R[9, 8, 3] = -100


env = gym.make("matrix_mdp/MatrixMDP-v0", p_0=p_0, p=P, r=R)

#################################################################
# Helper Functions
#################################################################


# reverse map observations in 0-15 to (i,j)
def reverse_map(observation):
    return observation // 4, observation % 4


#################################################################
# Q-Learning
#################################################################


"""
Implementing a function for Q-learning with epsilon-greedy exploration.
"""


def valid_action(state, action):
    # Getting coords for of state
    if state < 4:
        i, j = 0, state
    if 3 < state < 8:
        i, j = 1, state - 4
    if 7 < state < 12:
        i, j = 2, state - 8
    if 11 < state:
        i, j = 3, state - 12
    n = valid_neighbors(i, j)
    if action in n:
        return True
    else:
        return False


def q_learning(num_episodes, checkpoints, gamma=0.9, epsilon=0.9):
    """
    Q-learning algorithm.

    Parameters:
    - num_episodes (int): Number of Q-value episodes to perform.
    - checkpoints (list): List of episode numbers at which to record the optimal value function..

    Returns:
    - Q (numpy array): Q-values of shape (nS, nA) after all episodes.
    - optimal_policy (numpy array): Optimal policy, np array of shape (nS,), ordered by state index.
    - V_opt_checkpoint_values (list of numpy arrays): A list of optimal value function arrays at specified episode numbers.
      The saved values at each checkpoint should be of shape (nS,).
    """

    Q = np.zeros((nS, nA), dtype=np.float64)
    num_updates = np.zeros((nS, nA), dtype=np.float64)

    observation, info = env.reset()

    V_opt_checkpoint_values = []
    optimal_policy = np.zeros(nS)

    for episode in range(num_episodes):
        observation, info = env.reset()
        terminated, truncated = False, False

        while not (terminated or truncated):
            if np.random.rand() < epsilon:
                action = env.action_space.sample()

            else:
                action = np.argmax(Q[observation])
            while not valid_action(observation, action):
                action = env.action_space.sample()

            new_observation, reward, terminated, truncated, info = env.step(action)

            max_next_q = np.max(Q[new_observation])
            eta = 1 / (1 + num_updates[observation, action])
            print("reward is:" + str(reward))
            print("Max next q is:" + str(max_next_q))
            print("Q[Observation, action] is:" + str(Q[observation, action]))
            print("state is:" + str(observation))
            print("action is:" + str(action))

            Q[observation, action] += eta * (
                reward + gamma * max_next_q - Q[observation, action]
            )
            num_updates[observation, action] += 1

            observation = new_observation

        # Decay epsilon
        epsilon *= 0.9999

        # Checkpoint the V_opt values
        if episode + 1 in checkpoints:
            V_opt = np.max(Q, axis=1)
            V_opt_checkpoint_values.append(V_opt)

    optimal_policy = np.argmax(Q, axis=1)
    optimal_policy[2] = -1  # Hard coding for end states
    optimal_policy[13] = -1  # Hard coding for end states

    print(V_opt)
    print(optimal_policy)
    return Q, optimal_policy, V_opt_checkpoint_values


def plot_heatmaps(V_opt, filename):
    """
    Plots a 4x4 heatmap of the optimal value function, with state positions
    corresponding to cells in the map of Mordor, with the given filename.

    Parameters:
    V_opt (numpy array): A numpy array of shape (nS,) representing the optimal value function.
    filename (str): The filename to save the plot to.

    Returns:
    None
    """
    grid = V_opt.reshape((4, 4))
    plt.figure(figsize=(8, 6))
    sns.heatmap(grid, annot=True, cmap="YlGnBu", cbar=True, square=True)
    plt.savefig(filename)
    plt.close()




def main():

    Q, optimal_policy, V_opt_checkpoint_values = q_learning(
        10000, checkpoints=[10, 500, 10000]
    )
    plot_heatmaps(V_opt_checkpoint_values[0], "heatmap_10.png")
    plot_heatmaps(V_opt_checkpoint_values[1], "heatmap_500.png")
    plot_heatmaps(V_opt_checkpoint_values[1], "heatmap_10000.png")


if __name__ == "__main__":
    main()
