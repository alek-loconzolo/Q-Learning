# Q-Learning on a Grid World Environment (Matrix MDP)

This repository implements Q-Learning for a 4x4 grid world environment (called "Matrix MDP"). The agent moves in a grid and receives rewards or penalties for reaching certain states. This project utilizes Gymnasium's custom `MatrixMDP` environment to run and optimize the Q-learning algorithm. The project includes visualizing the optimal value function at different stages of training using heatmaps.

## Dependencies

Before running the code, ensure you have the following libraries installed:

- `numpy`
- `tqdm`
- `gymnasium`
- `matplotlib`
- `seaborn`

You can install these using the following:

```bash
pip install numpy tqdm gymnasium matplotlib seaborn
```

## Environment Description

The environment is a 4x4 grid (16 states), and the agent can move in four directions: up, down, left, and right. The movement is stochastic, with a slip probability that may cause the agent to take a random action different from the intended one.

The rewards and penalties are:
- Reaching a specific state provides high rewards (e.g., +2000).
- Some states incur penalties (e.g., -100) when reached.
- The agent can also experience penalties for slipping into certain states.

## Project Structure

1. **Grid Setup**:
   - The state transition probabilities (`P`) and reward matrix (`R`) are manually defined. The slip probability introduces stochasticity into the agent's movements.

2. **Q-Learning Algorithm**:
   The Q-Learning algorithm is implemented with epsilon-greedy exploration. The agent updates the Q-values based on the reward received and the future state’s maximum Q-value.
   - The learning rate (`eta`) is adjusted based on the number of updates.
   - Epsilon decreases with each episode to encourage the agent to explore early and exploit later.

3. **Checkpointing**:
   The optimal value function is saved at different checkpoints (`10`, `500`, and `10000` episodes) to monitor the learning progress.

4. **Visualization**:
   Heatmaps are generated for the optimal value function at specified checkpoints. The heatmaps display the optimal state values in the 4x4 grid.

## Key Functions

### 1. `q_learning`
This function implements Q-Learning with epsilon-greedy exploration and performs updates to the Q-values based on the agent's experience.

#### Parameters:
- `num_episodes`: Total number of episodes for Q-Learning.
- `checkpoints`: List of episodes at which the optimal value function is saved.
- `gamma`: Discount factor for future rewards (default is `0.9`).
- `epsilon`: Initial epsilon value for exploration (default is `0.9`).

#### Returns:
- `Q`: The learned Q-values.
- `optimal_policy`: The learned optimal policy (best action for each state).
- `V_opt_checkpoint_values`: A list of optimal value function arrays at specified episode numbers.

### 2. `plot_heatmaps`
This function generates and saves heatmaps of the optimal value function in a 4x4 grid.

#### Parameters:
- `V_opt`: The optimal value function array.
- `filename`: The name of the file to save the heatmap.

## How to Run the Code

1. Clone the repository and navigate to the directory.
2. Run the main file:

```bash
python q_learning_gridworld.py
```

The code will execute Q-Learning for 10,000 episodes and generate the following files:

- `heatmap_10.png`: Heatmap after 10 episodes.
- `heatmap_500.png`: Heatmap after 500 episodes.
- `heatmap_10000.png`: Heatmap after 10,000 episodes.

## Results and Visualization

As the agent learns, you will observe how the value function evolves through the checkpoints. The heatmaps display the optimal value function for each state in the 4x4 grid.

### Example Output:

- **Heatmap after 10 Episodes**: Shows the initial exploration phase where the agent hasn't fully learned the environment.
- **Heatmap after 500 Episodes**: The agent has started learning and improving its policy.
- **Heatmap after 10,000 Episodes**: The agent has converged to a near-optimal policy, with high-value states corresponding to the rewarding states.
