import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import matplotlib.pyplot as plt
from collections import deque

# -------------------------------------------------
# 1. Función para convertir el estado discreto a vector one-hot
# -------------------------------------------------
def one_hot(state, state_size):
    vec = np.zeros(state_size, dtype=np.float32)
    vec[state] = 1.0
    return vec

# -------------------------------------------------
# 2. Wrapper para el entorno Frozen Lake
# -------------------------------------------------
class FrozenLakeWrapper:
    def __init__(self, is_slippery=False, map_name="4x4"):
        self.env = gym.make('FrozenLake-v1', is_slippery=is_slippery, map_name=map_name)
        self.action_space = self.env.action_space  
        self.observation_space = self.env.observation_space  
    
    def reset(self, seed=None):
        # Gymnasium devuelve (observation, info)
        result = self.env.reset(seed=seed)
        if isinstance(result, tuple):
            return result[0]
        return result
    
    def step(self, action):
        result = self.env.step(action)
        if len(result) == 5:
            obs, reward, terminated, truncated, info = result
            done = terminated or truncated
            return obs, reward, done, info
        else:
            return result
    
    def render(self):
        self.env.render()

# -------------------------------------------------
# 3. Definición de la red neuronal para SARSA
# -------------------------------------------------
class SarsaNet(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=64):
        super(SarsaNet, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, action_size)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# -------------------------------------------------
# 4. Agente SARSA con aproximación semigradiente
# -------------------------------------------------
class SarsaSemigrad:
    def __init__(self, state_size, action_size, hidden_size=64, lr=0.001, gamma=0.99,
                 epsilon=1.0, min_epsilon=0.1, decay_rate=0.995):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.decay_rate = decay_rate
        
        self.model = SarsaNet(state_size, action_size, hidden_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
    
    def choose_action(self, state):
        # Política epsilon-greedy
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.model(state_tensor)
            return q_values.argmax().item()
    
    def update(self, state, action, reward, next_state, next_action, done):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
        
        # Q(s, a) actual
        q_value = self.model(state_tensor)[0, action]
        
        with torch.no_grad():
            if done:
                target = reward
            else:
                target = reward + self.gamma * self.model(next_state_tensor)[0, next_action]
        target = torch.tensor(target)
        
        loss = self.criterion(q_value, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    def decay_epsilon(self):
        self.epsilon = max(self.epsilon * self.decay_rate, self.min_epsilon)

# -------------------------------------------------
# 5. Función de entrenamiento del agente SARSA
# -------------------------------------------------
def train_sarsa(env_wrapper, agent, num_episodes=10000, seed=1995):
    state_size = env_wrapper.observation_space.n
    episode_rewards = []      # Recompensa total por episodio
    episode_lengths = []      # Número de pasos por episodio
    
    for episode in range(num_episodes):
        state = env_wrapper.reset(seed=seed)
        state_one_hot = one_hot(state, state_size)
        action = agent.choose_action(state_one_hot)
        total_reward = 0
        steps = 0
        done = False
        
        while not done:
            next_state, reward, done, info = env_wrapper.step(action)
            next_state_one_hot = one_hot(next_state, state_size)
            next_action = agent.choose_action(next_state_one_hot) if not done else None
            
            # Actualización en línea: semi-gradient SARSA
            agent.update(state_one_hot, action, reward, next_state_one_hot, next_action, done)
            
            state_one_hot = next_state_one_hot
            action = next_action if not done else None
            total_reward += reward
            steps += 1
        
        agent.decay_epsilon()
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        
        if (episode + 1) % 50 == 0:
            avg_reward = np.mean(episode_rewards[-50:])
            print(f"Episode {episode+1}, Average Reward: {avg_reward:.2f}, Epsilon: {agent.epsilon:.2f}")
            
    return episode_rewards, episode_lengths, agent

# -------------------------------------------------
# 6. Clase para visualización de estadísticas (opcional)
# -------------------------------------------------
class GraphVisualizer:
    def __init__(self, rewards, lengths, rolling_length=500):
        self.rewards = rewards
        self.lengths = lengths
        self.rolling_length = rolling_length

    def get_moving_avgs(self, arr, window, mode):
        return np.convolve(np.array(arr).flatten(), np.ones(window), mode=mode) / window

    def plot_all(self):
        fig, axs = plt.subplots(ncols=2, figsize=(12, 5))
        
        # Promedio móvil de recompensas por episodio
        axs[0].set_title("Episode Rewards")
        reward_moving_avg = self.get_moving_avgs(self.rewards, self.rolling_length, "valid")
        axs[0].plot(range(len(reward_moving_avg)), reward_moving_avg)
        axs[0].set_xlabel("Episodes")
        axs[0].set_ylabel("Reward")
        
        # Promedio móvil de longitudes de episodio
        axs[1].set_title("Episode Lengths")
        length_moving_avg = self.get_moving_avgs(self.lengths, self.rolling_length, "valid")
        axs[1].plot(range(len(length_moving_avg)), length_moving_avg)
        axs[1].set_xlabel("Episodes")
        axs[1].set_ylabel("Length")
        
        plt.tight_layout()
        plt.show()

