import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt

# -------------------------------------------------
# 1. Definición de la red neuronal (DQN)
# -------------------------------------------------
class DQN(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=64):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, action_size)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# -------------------------------------------------
# 2. Replay Buffer para almacenar transiciones
# -------------------------------------------------
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return np.array(states), actions, rewards, np.array(next_states), dones
    
    def __len__(self):
        return len(self.buffer)

# -------------------------------------------------
# 3. Función para convertir un estado discreto en vector one-hot
# -------------------------------------------------
def one_hot(state, state_size):
    vec = np.zeros(state_size, dtype=np.float32)
    vec[state] = 1.0
    return vec

# -------------------------------------------------
# 4. Función de entrenamiento del agente DQN
# -------------------------------------------------
def train_dqn(env, num_episodes=1000, batch_size=64, target_update=10, seed=1995):
    # Configurar la semilla para el entorno en cada reset usando el parámetro 'seed'
    state_size = env.observation_space.n
    action_size = env.action_space.n
    
    # Hiperparámetros
    hidden_size = 64
    learning_rate = 0.001
    gamma = 0.99

    epsilon = 1.0
    min_epsilon = 0.01
    epsilon_decay = 0.995

    buffer_capacity = 10000

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Inicialización de redes y optimizador
    policy_net = DQN(state_size, action_size, hidden_size).to(device)
    target_net = DQN(state_size, action_size, hidden_size).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)
    replay_buffer = ReplayBuffer(buffer_capacity)

    episode_rewards = []
    episode_lengths = []      # Si deseas almacenar la longitud de cada episodio
    training_errors = []      # Para almacenar el error en cada actualización

    for episode in range(num_episodes):
        # Aquí usamos el parámetro 'seed' para reinicializar el entorno
        state, _ = env.reset(seed=seed)
        state = one_hot(state, state_size)
        total_reward = 0
        done = False
        step_count = 0

        while not done:
            step_count += 1

            # Selección de acción: política epsilon-greedy
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    state_tensor = torch.tensor(state, device=device).unsqueeze(0)
                    q_values = policy_net(state_tensor)
                    action = q_values.argmax().item()
            
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_state_one_hot = one_hot(next_state, state_size)
            
            replay_buffer.push(state, action, reward, next_state_one_hot, done)
            state = next_state_one_hot
            total_reward += reward
            
            if len(replay_buffer) >= batch_size:
                states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
                
                states = torch.tensor(states, device=device)
                actions = torch.tensor(actions, device=device).unsqueeze(1)
                rewards = torch.tensor(rewards, device=device)
                next_states = torch.tensor(next_states, device=device)
                dones = torch.tensor(dones, device=device, dtype=torch.float32)
                
                q_values = policy_net(states).gather(1, actions).squeeze(1)
                with torch.no_grad():
                    max_next_q_values = target_net(next_states).max(1)[0]
                    target = rewards + gamma * max_next_q_values * (1 - dones)
                
                loss = nn.MSELoss()(q_values, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                training_errors.append(loss.item())
        
        episode_rewards.append(total_reward)
        episode_lengths.append(step_count)
        epsilon = max(epsilon * epsilon_decay, min_epsilon)
        
        if (episode + 1) % 50 == 0:
            avg_reward = np.mean(episode_rewards[-50:])
            print(f"Episode {episode+1}, Average Reward: {avg_reward:.2f}, Epsilon: {epsilon:.2f}")
        
        if episode % target_update == 0:
            target_net.load_state_dict(policy_net.state_dict())

    return episode_rewards, episode_lengths, training_errors, policy_net, target_net

# -------------------------------------------------
# 5. Clase para visualización de estadísticas
# -------------------------------------------------
class GraphVisualizer:
    def __init__(self, rewards, lengths, training_errors, rolling_length=500):
        self.rewards = rewards
        self.lengths = lengths
        self.training_errors = training_errors
        self.rolling_length = rolling_length

    def get_moving_avgs(self, arr, window, convolution_mode):
        return np.convolve(
            np.array(arr).flatten(),
            np.ones(window),
            mode=convolution_mode
        ) / window

    def plot_all(self):
        fig, axs = plt.subplots(ncols=2, figsize=(12, 5))

        axs[0].set_title("Episode Rewards")
        reward_moving_average = self.get_moving_avgs(self.rewards, self.rolling_length, "valid")
        axs[0].plot(range(len(reward_moving_average)), reward_moving_average)
        axs[0].set_xlabel("Episodios")
        axs[0].set_ylabel("Recompensa")

        axs[1].set_title("Episode Lengths")
        length_moving_average = self.get_moving_avgs(self.lengths, self.rolling_length, "valid")
        axs[1].plot(range(len(length_moving_average)), length_moving_average)
        axs[1].set_xlabel("Episodios")
        axs[1].set_ylabel("Longitud")

        
        plt.tight_layout()
        plt.show()


