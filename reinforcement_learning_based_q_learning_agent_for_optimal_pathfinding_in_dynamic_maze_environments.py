import numpy as np
import matplotlib.pyplot as plt

class SimpleMaze:
    def __init__(self, layout, start, goal):
        self.layout = layout
        self.height = layout.shape[0]
        self.width = layout.shape[1]
        self.start = start
        self.goal = goal

    def display(self):
        plt.figure(figsize=(5, 5))
        plt.imshow(self.layout, cmap='gray')
        plt.text(self.start[1], self.start[0], 'S', ha='center', va='center', color='red', fontsize=20)
        plt.text(self.goal[1], self.goal[0], 'G', ha='center', va='center', color='green', fontsize=20)
        plt.xticks([]), plt.yticks([])
        plt.grid(color='black', linewidth=2)
        plt.show()

moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, Down, Left, Right

class QLearningNavigator:
    def __init__(self, maze, lr=0.1, gamma=0.9, epsilon_start=1.0, epsilon_end=0.01, episodes=100):
        self.q_values = np.zeros((maze.height, maze.width, 4))  # 4 actions: Up, Down, Left, Right
        self.learning_rate = lr
        self.discount_factor = gamma
        self.exploration_start = epsilon_start
        self.exploration_end = epsilon_end
        self.episodes = episodes

    def calculate_exploration_rate(self, episode):
        rate = self.exploration_start * (self.exploration_end / self.exploration_start) ** (episode / self.episodes)
        return rate

    def select_action(self, state, episode):
        rate = self.calculate_exploration_rate(episode)
        if np.random.rand() < rate:
            return np.random.randint(4)
        else:
            return np.argmax(self.q_values[state])

    def update_q_values(self, state, action, next_state, reward):
        best_next_action = np.argmax(self.q_values[next_state])
        current_q = self.q_values[state][action]
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * self.q_values[next_state][best_next_action] - current_q)
        self.q_values[state][action] = new_q

def run_episode(navigator, maze, episode, training=True):
    state = maze.start
    done = False
    total_reward = 0
    steps = 0
    path = [state]

    while not done:
        action = navigator.select_action(state, episode)
        next_state = (state[0] + moves[action][0], state[1] + moves[action][1])

        if next_state[0] < 0 or next_state[0] >= maze.height or next_state[1] < 0 or next_state[1] >= maze.width or maze.layout[next_state[0], next_state[1]] == 1:
            reward = -10
            next_state = state
        elif next_state == maze.goal:
            path.append(next_state)
            reward = 100
            done = True
        else:
            path.append(next_state)
            reward = -1

        total_reward += reward
        steps += 1

        if training:
            navigator.update_q_values(state, action, next_state, reward)

        state = next_state

    return total_reward, steps, path

def evaluate_navigator(navigator, maze, eval_episodes=100):
    reward, steps, path = run_episode(navigator, maze, eval_episodes, training=False)

    print("Path taken:")
    for r, c in path:
        print(f"({r}, {c})-> ", end='')
    print("Goal!")

    print("Steps taken:", steps)
    print("Total reward:", reward)

    if plt.gcf().get_axes():
        plt.cla()

    plt.figure(figsize=(5, 5))
    plt.imshow(maze.layout, cmap='copper')
    plt.text(maze.start[1], maze.start[0], 'o', ha='center', va='center', color='red', fontsize=20)
    plt.text(maze.goal[1], maze.goal[0], 'o', ha='center', va='center', color='green', fontsize=20)
    for pos in path:
        plt.text(pos[1], pos[0], ".", va='center', color='blue', fontsize=20)
    plt.xticks([]), plt.yticks([])
    plt.grid(color='black', linewidth=2)
    plt.show()

    return steps, reward

def train_navigator(navigator, maze, total_episodes=100):
    rewards_per_episode = []
    steps_per_episode = []

    for ep in range(total_episodes):
        reward, steps, _ = run_episode(navigator, maze, ep, training=True)
        rewards_per_episode.append(reward)
        steps_per_episode.append(steps)

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.plot(rewards_per_episode)
    plt.xlabel('Episode')
    plt.ylabel('Cumulative Reward')
    plt.title('Reward per Episode')

    avg_reward = sum(rewards_per_episode) / len(rewards_per_episode)
    print(f"Average reward: {avg_reward}")

    plt.subplot(1, 2, 2)
    plt.plot(steps_per_episode)
    plt.xlabel('Episode')
    plt.ylabel('Steps Taken')
    plt.ylim(0, 100)
    plt.title('Steps per Episode')

    avg_steps = sum(steps_per_episode) / len(steps_per_episode)
    print(f"Average steps: {avg_steps}")

    plt.tight_layout()
    plt.show()

# Define 12 by 12 matrix :
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

# Define starting and ending positions in matrix
maze = SimpleMaze(maze_layout, (0, 0), (11, 11))
navigator = QLearningNavigator(maze)

# Training the agents to update Qtable
train_navigator(navigator, maze, total_episodes=100)

# Testing accuracy of the model
evaluate_navigator(navigator, maze)