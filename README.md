# Implementation-of-SARSA-Control-Algorithm-using-Gymnasium

## Aim

To implement the **SARSA control algorithm** using the Gymnasium `FrozenLake-v1` environment and learn an action-value function that helps the agent select better actions for reaching the goal state while avoiding holes.

---

## Problem Statement

To implement the State-Action-Reward-State-Action (SARSA) control algorithm using the Gymnasium FrozenLake environment.

The objective is to train an agent to navigate through a 4×4 frozen lake grid, starting from the initial state and reaching the goal state while avoiding holes. The agent learns the best actions through trial and error using rewards received from the environment.

## Software Requirements
- Python 3.x
- Jupyter Notebook or Google Colab
- Gymnasium
- NumPy
- Matplotlib


## Environment Description
The experiment uses the FrozenLake-v1 environment provided by Gymnasium.

The environment consists of a 4×4 grid containing four types of tiles:

| Symbol | Description |
|---|---|
| S | Starting position |
| F | Frozen surface (safe) |
| H | Hole (terminal state) |
| G | Goal state (terminal state) |


## Theory

SARSA stands for:

$$
S_t, A_t, R_{t+1}, S_{t+1}, A_{t+1}
$$

It updates the Q-value using the action actually selected in the next state.

The SARSA update rule is:

$$
Q(S_t,A_t) \leftarrow Q(S_t,A_t) + \alpha
\left[
R_{t+1} + \gamma Q(S_{t+1},A_{t+1}) - Q(S_t,A_t)
\right]
$$

Where:

| Symbol | Meaning |
|---|---|
| $S_t$ | Current state |
| $A_t$ | Current action |
| $R_{t+1}$ | Reward received after taking action $A_t$ |
| $S_{t+1}$ | Next state |
| $A_{t+1}$ | Next action selected using the current policy |
| $\alpha$ | Learning rate |
| $\gamma$ | Discount factor |
| $Q(s,a)$ | Action-value function |

---

## Epsilon-Greedy Policy

SARSA uses an epsilon-greedy policy for action selection.

With probability $\epsilon$, the agent explores by selecting a random action.

With probability $1-\epsilon$, the agent exploits by selecting the action with the highest Q-value.

$$
a =
\begin{cases}
\text{random action}, & \text{with probability } \epsilon \\
\arg\max_a Q(s,a), & \text{with probability } 1-\epsilon
\end{cases}
$$

---


## Algorithm


## Python Program

```python

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
env = gym.make("FrozenLake-v1", is_slippery=True)

n_states = env.observation_space.n
n_actions = env.action_space.n
# -------------------------------------------------
# Hyperparameters
# -------------------------------------------------

num_episodes = 10000
max_steps_per_episode = 100

alpha = 0.1          # Learning rate
gamma = 0.99         # Discount factor

epsilon = 1.0        # Initial exploration rate
epsilon_min = 0.05
epsilon_decay = 0.9995
Q = np.zeros((n_states, n_actions))


# -------------------------------------------------
# Epsilon-Greedy Action Selection
# -------------------------------------------------

def epsilon_greedy_action(state, epsilon):
    """
    Selects an action using epsilon-greedy strategy.
    """
    if np.random.random() < epsilon:
        return env.action_space.sample()
    else:
        return int(np.argmax(Q[state]))
# -------------------------------------------------
# SARSA Training
# -------------------------------------------------

episode_rewards = []

for episode in range(num_episodes):

    state, info = env.reset()

    action = epsilon_greedy_action(state, epsilon)

    total_reward = 0

    for step in range(max_steps_per_episode):

        next_state, reward, terminated, truncated, info = env.step(action)

        total_reward += reward

        if terminated:
            Q[state, action] += alpha * (
                reward - Q[state, action]
            )
            break

        next_action = epsilon_greedy_action(next_state, epsilon)

        target = reward + gamma * Q[next_state, next_action]

        Q[state, action] += alpha * (
            target - Q[state, action]
        )

        state = next_state
        action = next_action

        if truncated:
            break

    episode_rewards.append(total_reward)

    epsilon = max(
        epsilon_min,
        epsilon * epsilon_decay
    )


# -------------------------------------------------
# Display Functions
# -------------------------------------------------

def print_value_function(values):
    print("\nEstimated State-Value Function:")
    print(np.round(values.reshape(4, 4), 3))


def print_policy(policy):
    action_symbols = {
        0: "L",
        1: "D",
        2: "R",
        3: "U"
    }

    policy_grid = np.array(
        [action_symbols[action] for action in policy]
    ).reshape(4, 4)

    print("\nLearned Policy:")
    print(policy_grid)


# -------------------------------------------------
# Output
# -------------------------------------------------
state_values = np.max(Q, axis=1)

learned_policy = np.argmax(Q, axis=1)
print("\nFinal Q-table:")
print(np.round(Q, 3))
print("NAME:SAI GURU CHANDRAN G\nREGNO:212223240143")
print_value_function(state_values)
print_policy(learned_policy)

average_reward = np.mean(episode_rewards[-1000:])
print("\nAverage reward over last 1000 episodes:", average_reward)
    



```
---

## Output

```text

Final Q-table:
[[0.412 0.335 0.336 0.344]
 [0.255 0.221 0.246 0.327]
 [0.257 0.264 0.243 0.26 ]
 [0.191 0.194 0.192 0.251]
 [0.434 0.222 0.161 0.178]
 [0.    0.    0.    0.   ]
 [0.209 0.147 0.16  0.087]
 [0.    0.    0.    0.   ]
 [0.29  0.318 0.383 0.495]
 [0.264 0.545 0.403 0.361]
 [0.557 0.374 0.227 0.233]
 [0.    0.    0.    0.   ]
 [0.    0.    0.    0.   ]
 [0.367 0.515 0.665 0.292]
 [0.662 0.845 0.697 0.677]
 [0.    0.    0.    0.   ]]
NAME:SAI GURU CHANDRAN G
REGNO:212223240143

Estimated State-Value Function:
[[0.412 0.327 0.264 0.251]
 [0.434 0.    0.209 0.   ]
 [0.495 0.545 0.557 0.   ]
 [0.    0.665 0.845 0.   ]]

Learned Policy:
[['L' 'U' 'D' 'U']
 ['L' 'L' 'L' 'L']
 ['U' 'D' 'L' 'L']
 ['L' 'R' 'D' 'L']]

Average reward over last 1000 episodes: 0.509

```

---
<img width="831" height="532" alt="image" src="https://github.com/user-attachments/assets/bc828539-9d4c-4168-9492-651969bc2468" />

## Result
```text

The SARSA control algorithm was successfully implemented using the Gymnasium FrozenLake environment. The agent learned to select suitable actions through repeated interactions with the environment. After training, the final Q-table, estimated state-value function, and learned policy were generated. The learning curve was also plotted to observe the agent's improvement during training.

```

---

## Inference
```text



From this experiment, we understood how the SARSA algorithm learns an optimal action-selection strategy through trial and error. The epsilon-greedy strategy helps the agent balance exploration and exploitation. Since SARSA is an on-policy algorithm, it updates Q-values using the next action selected by its current policy. The experiment demonstrates how reinforcement learning can be used to make decisions in an uncertain environment.

```
---

