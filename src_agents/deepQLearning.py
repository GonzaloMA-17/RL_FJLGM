import os
import gc
import torch
import numpy as np
import gymnasium as gym
import random
from collections import deque
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim

# --- Configuración para reproducibilidad ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
seed = 2024  # Cambia este valor para probar con otra semilla
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

# --- Definición de la red neuronal (DQN) ---
class DQN(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=64):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, action_size)
        nn.init.kaiming_uniform_(self.fc1.weight, nonlinearity='relu')
        nn.init.kaiming_uniform_(self.fc2.weight, nonlinearity='linear')
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# --- Replay Buffer ---
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size, rng=None):
        if rng is None:
            batch = random.sample(list(self.buffer), batch_size)
        else:
            batch = rng.sample(list(self.buffer), batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return np.array(states), actions, rewards, np.array(next_states), dones
    
    def __len__(self):
        return len(self.buffer)

# --- Función para convertir estado a one-hot ---
def one_hot(state, state_size):
    vec = np.zeros(state_size, dtype=np.float32)
    vec[state] = 1.0
    return vec

# --- Función para sampling con generador específico (alternativa) ---
def sample_buffer(buffer, batch_size, rng):
    batch = rng.sample(list(buffer.buffer), batch_size)
    states, actions, rewards, next_states, dones = zip(*batch)
    return np.array(states), actions, rewards, np.array(next_states), dones

# --- Función de entrenamiento del agente DQN ---
def train_dqn(env, num_episodes=1000, batch_size=64, target_update=10, seed=2024):
    # Establecer todas las semillas necesarias
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    state_size = env.observation_space.n
    action_size = env.action_space.n
    
    hidden_size = 64
    learning_rate = 0.001
    gamma = 0.99

    epsilon = 1.0
    min_epsilon = 0.01
    epsilon_decay = 0.995

    buffer_capacity = 10000

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Asegurar determinismo en PyTorch
    if torch.cuda.is_available():
        torch.use_deterministic_algorithms(True)

    policy_net = DQN(state_size, action_size, hidden_size).to(device)
    target_net = DQN(state_size, action_size, hidden_size).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)
    replay_buffer = ReplayBuffer(buffer_capacity)

    episode_rewards = []
    episode_lengths = []
    training_errors = []
    
    # Generador de números aleatorios con semilla fija para el buffer
    rng = random.Random(seed)

    for episode in range(num_episodes):
        # Reset con semilla específica por episodio (pero determinista)
        episode_seed = seed + episode  # Cada episodio tiene su propia semilla derivada
        state, _ = env.reset(seed=episode_seed)
        state = one_hot(state, state_size)
        total_reward = 0
        done = False
        step_count = 0

        while not done:
            step_count += 1
            
            # Usar el mismo generador para decisiones epsilon-greedy
            if rng.random() < epsilon:
                # Usar nuestro rng para muestrear acción
                action = rng.randint(0, action_size - 1)
            else:
                with torch.no_grad():
                    state_tensor = torch.tensor(state, device=device).unsqueeze(0)
                    q_values = policy_net(state_tensor)
                    action = q_values.argmax().item()
            
            # Paso en el entorno
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_state_one_hot = one_hot(next_state, state_size)
            
            replay_buffer.push(state, action, reward, next_state_one_hot, done)
            state = next_state_one_hot
            total_reward += reward
            
            if len(replay_buffer) >= batch_size:
                # Usar nuestro generador para el muestreo
                states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size, rng)
                
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

