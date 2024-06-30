# Reinforcement-Learning-based-Q-Learning-Agent-for-Optimal-Pathfinding-in-Dynamic-Maze-Environments
This project involves reinforcement learning (specifically Q-learning) and focuses on navigating through a maze environment to find solutions.
The agent learns to find the optimal path from a starting position to a goal position using reinforcement learning principles.

## Project Overview

The project consists of Python code that simulates a simple maze environment and implements the Q-learning algorithm. 

- **Maze**: Defines the maze layout, including walls and open paths. It also marks the starting and goal positions.

- **QLearningNavigator**: Implements the Q-learning algorithm. It initializes a Q-table to store state-action values and includes methods for selecting actions, updating Q-values, and controlling exploration-exploitation trade-off.

- **Training and Evaluation**: Includes functions to train the Q-learning agent over multiple episodes and to evaluate its performance by navigating the maze.

## Algorithm Explanation

### Q-Learning Algorithm

Q-learning is a model-free reinforcement learning algorithm that learns an optimal policy for an agent in an environment.

1. **Q-Table Initialization**: Initialize a Q-table with zeros. Each cell in the table represents a state-action pair, where Q(s, a) denotes the expected cumulative reward for taking action 'a' from state 's'.

2. **Exploration vs. Exploitation**: During training, the agent selects actions based on an exploration-exploitation trade-off. Initially, it explores random actions to discover new paths (exploration). As training progresses, it increasingly exploits learned Q-values to maximize rewards.

3. **Q-Value Update**: When the agent takes an action and transitions to a new state:

![image](https://github.com/NipunaMuhandiram/Reinforcement-Learning-based-Q-Learning-Agent-for-Optimal-Pathfinding-in-Dynamic-Maze-Environments/assets/75882756/ecef1b18-f6f3-46a0-9294-d1fde5014681)

   - $( \alpha )$ is the learning rate, controlling how much new information overrides old information.
   - $( \gamma )$ is the discount factor, balancing immediate and future rewards.
   - $( r )$ is the reward received for taking action 'a' from state 's'.

4. **Training**: The agent iteratively interacts with the environment, updating Q-values based on observed rewards, aiming to learn the optimal policy that leads from the start to the goal.

## Mathematical Background

### Q-Learning Update Equation

The Q-value update equation is derived from the Bellman equation for optimality in reinforcement learning:

![image](https://github.com/NipunaMuhandiram/Reinforcement-Learning-based-Q-Learning-Agent-for-Optimal-Pathfinding-in-Dynamic-Maze-Environments/assets/75882756/ecef1b18-f6f3-46a0-9294-d1fde5014681)

Where:
- $Q(s,a)$ is the Q-value for taking action 'a' in state 's'.
- $( \alpha )$ is the learning rate, controlling the step size of updates.
- $( r )$ is the reward received after taking action 'a' in state 's'.
- $( \gamma )$ is the discount factor, determining the importance of future rewards.
- $( s' )$ is the next state after taking action 'a'.

### Exploration-Exploitation

Balancing exploration and exploitation is crucial in Q-learning:
- **Exploration**: Allows the agent to discover new paths and potentially better policies.
- **Exploitation**: Uses learned knowledge to maximize rewards in the current policy.
  
## Results and Visualization

The project demonstrates the effectiveness of Q-learning in solving dynamic pathfinding tasks within maze environments. Results include visual representations of learned paths and performance metrics such as cumulative rewards and steps per episode.
### Training Results
![image](https://github.com/NipunaMuhandiram/Reinforcement-Learning-based-Q-Learning-Agent-for-Optimal-Pathfinding-in-Dynamic-Maze-Environments/assets/75882756/12157886-f4a0-4dfb-b574-b72201bea804)
![image](https://github.com/NipunaMuhandiram/Reinforcement-Learning-based-Q-Learning-Agent-for-Optimal-Pathfinding-in-Dynamic-Maze-Environments/assets/75882756/05e8baac-0c9d-436d-9f18-ff1b2691eab0)


### Evaluation Results
![image](https://github.com/NipunaMuhandiram/Reinforcement-Learning-based-Q-Learning-Agent-for-Optimal-Pathfinding-in-Dynamic-Maze-Environments/assets/75882756/a3a5bd7e-d182-43eb-935b-eef7d17b031c)

## Usage

1. **Environment Setup**: Define your maze layout using a 2D numpy array, where `0` represents open paths and `1` represents walls.

2. **Initialization**: Create a `SimpleMaze` object with the maze layout and specify the start and goal positions.

3. **Training**: Instantiate a `QLearningNavigator` object with the maze object. Use the `train_navigator` function to train the agent over a specified number of episodes.

4. **Evaluation**: Use the `evaluate_navigator` function to test the trained agent's performance. It will display the path taken, steps, and total reward achieved.

```python
import numpy as np
import matplotlib.pyplot as plt

# Define your maze layout here
maze_layout = np.array([
    [0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0],
    [0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0],
    [1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0],
    [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    [1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1],
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0],
    [0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0],
    [0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
    [1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0]
])

# Initialize maze and navigator
maze = SimpleMaze(maze_layout, (0, 0), (11, 11))
navigator = QLearningNavigator(maze)

# Train the agent
train_navigator(navigator, maze, total_episodes=100)

# Evaluate the agent
evaluate_navigator(navigator, maze)
```
## Credits
This project utilizes open-source contributions from the following libraries:
- [NumPy](https://numpy.org): For efficient numerical operations in Python.
- [Matplotlib](https://matplotlib.org): For visualization of maze environments and learning progress.

### Open-Source Guides
- [R.L Python](https://opensource.com/article/17/11/reinforcement-learning): Getting started with reinforcement learning.
- [Q-Learning](https://www.datacamp.com/tutorial/introduction-q-learning-beginner-tutorial): An Introduction to Q-Learning: A Tutorial For Beginners.
- [Matplotlib Documentation](https://matplotlib.org/stable/contents.html): Comprehensive documentation for creating visualizations with Matplotlib.
- [NumPy Documentation](https://numpy.org/doc/stable/): Official documentation for NumPy, providing extensive support for numerical computations in Python.

