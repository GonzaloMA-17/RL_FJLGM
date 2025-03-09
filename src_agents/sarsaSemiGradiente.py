# import numpy as np
# import gymnasium as gym
# from tqdm import tqdm
# from typing import *
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from src_agents import Agente
# from .politicas import epsilon_greedy_policy

# class SarsaSemigradiente(Agente):
#     def __init__(self, 
#                  env: gym.Env, 
#                  epsilon: float = 0.4, 
#                  gamma: float = 0.99, 
#                  alpha: float = 0.1,
#                  decay: bool = False, 
#                  num_episodios: int = 5000,
#                  hidden_size: int = 128):
#         """
#         Inicializa el agente Sarsa Semigradiente.
        
#         Parámetros:
#         - env: Entorno de Gymnasium.
#         - epsilon: Tasa de exploración inicial.
#         - gamma: Factor de descuento.
#         - alpha: Tasa de aprendizaje para el optimizador.
#         - decay: Indica si se decae la exploración con cada episodio.
#         - num_episodios: Número total de episodios de entrenamiento.
#         - hidden_size: Tamaño de la capa oculta en la red neuronal.
#         """
#         # Llamamos al constructor de la clase base (Agente)
#         super().__init__(env, epsilon, gamma)
        
#         self.decay = decay
#         self.num_episodios = num_episodios
#         self.alpha = alpha
        
#         # Definir la aproximación de la función Q con una red neuronal
#         self.n_inputs = env.observation_space.n  # Tamaño del espacio de estados
#         self.n_outputs = env.action_space.n      # Tamaño del espacio de acciones
        
#         # Arquitectura de la red
#         self.model = nn.Sequential(
#             nn.Linear(self.n_inputs, hidden_size),
#             nn.ReLU(),
#             nn.Linear(hidden_size, self.n_outputs)
#         )
        
#         # Optimizador
#         self.optimizer = optim.SGD(self.model.parameters(), lr=alpha)
        
#         # Para almacenar estadísticas de entrenamiento
#         self.stats = 0.0
#         self.list_stats = []
#         self.episode_lengths = []
    
#     def state_to_tensor(self, state: int) -> torch.Tensor:
#         """
#         Convierte un índice de estado a un tensor one-hot.
#         """
#         state_tensor = torch.zeros(self.n_inputs)
#         state_tensor[state] = 1.0
#         return state_tensor.unsqueeze(0)  # Añadir dimensión de batch
    
#     def get_q_values(self, state: int) -> torch.Tensor:
#         """
#         Obtiene los valores Q para un estado dado usando la red neuronal.
#         """
#         state_tensor = self.state_to_tensor(state)
#         return self.model(state_tensor)
    
#     def seleccionar_accion(self, estado: int) -> int:
#         """
#         Selecciona una acción usando política epsilon-greedy.
#         """
#         with torch.no_grad():
#             q_values = self.get_q_values(estado).squeeze(0).numpy()
        
#         # Usar política epsilon-greedy con los valores Q aproximados
#         if np.random.random() < self.epsilon:
#             return self.env.action_space.sample()  # Exploración
#         else:
#             return np.argmax(q_values)  # Explotación
    
#     def update_q(self, state: int, action: int, reward: float, next_state: int, next_action: int) -> float:
#         """
#         Actualiza la aproximación de la función Q usando Sarsa Semigradiente.
        
#         Devuelve:
#         - loss: El error de la actualización.
#         """
#         # Calcular el valor Q actual
#         q_values = self.get_q_values(state)
#         current_q = q_values[0, action]
        
#         # Calcular el valor Q objetivo
#         with torch.no_grad():
#             next_q_values = self.get_q_values(next_state)
#             next_q = next_q_values[0, next_action]
#             target_q = reward + self.gamma * next_q
        
#         # Calcular la pérdida
#         loss = 0.5 * (target_q - current_q) ** 2
        
#         # Optimización
#         self.optimizer.zero_grad()
#         loss.backward()
#         self.optimizer.step()
        
#         return loss.item()

#     def entrenar(self) -> Tuple[nn.Module, List[float], List[int]]:
#         """
#         Entrena al agente utilizando Sarsa Semigradiente.
        
#         Devuelve:
#         - self.model: La red neuronal entrenada.
#         - self.list_stats: Lista con la recompensa media acumulada tras cada episodio.
#         - self.episode_lengths: Lista con la longitud de cada episodio.
#         """
#         for t in tqdm(range(self.num_episodios), desc="Entrenando", unit="episodio", ncols=100):
#             state, _ = self.env.reset()
#             action = self.seleccionar_accion(state)
#             done = False
#             episode_reward = 0.0
#             episode_length = 0
            
#             while not done:
#                 # Decaimiento opcional de epsilon
#                 if self.decay:
#                     self.epsilon = max(0.01, min(1.0, 1000.0 / (t + 1)))
                
#                 # Interactuar con el entorno
#                 next_state, reward, terminated, truncated, info = self.env.step(action)
                
#                 # Seleccionar la siguiente acción
#                 next_action = self.seleccionar_accion(next_state)
                
#                 # Actualizar la función Q
#                 loss = self.update_q(state, action, reward, next_state, next_action)
                
#                 state = next_state
#                 action = next_action
#                 done = terminated or truncated
#                 episode_reward += reward
#                 episode_length += 1
            
#             # Guardar la longitud del episodio
#             self.episode_lengths.append(episode_length)
            
#             # Actualizar estadísticas
#             self.stats += episode_reward
#             self.list_stats.append(self.stats / (t + 1))
        
#         return self.model, self.list_stats, self.episode_lengths
    
#     def guardar_modelo(self, ruta: str) -> None:
#         """
#         Guarda el modelo en la ruta especificada.
#         """
#         torch.save(self.model.state_dict(), ruta)
    
#     def cargar_modelo(self, ruta: str) -> None:
#         """
#         Carga el modelo desde la ruta especificada.
#         """
#         self.model.load_state_dict(torch.load(ruta))

import os
import gc
import torch
import numpy as np
import gymnasium as gym
import random
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
        # Inicialización de pesos para mayor reproducibilidad
        nn.init.kaiming_uniform_(self.fc1.weight, nonlinearity='relu')
        nn.init.kaiming_uniform_(self.fc2.weight, nonlinearity='linear')
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# -------------------------------------------------
# 4. Agente SARSA con aproximación semigradiente
# -------------------------------------------------
class SarsaSemigrad:
    def __init__(self, state_size, action_size, hidden_size=128, lr=0.005, gamma=0.99,
             epsilon=1.0, min_epsilon=0.1, decay_rate=0.999, seed=2024):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.decay_rate = decay_rate
        
        # Configurar semilla para decisiones aleatorias
        self.rng = random.Random(seed)
        
        # Configurar semilla para PyTorch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        
        self.model = SarsaNet(state_size, action_size, hidden_size)
        self.model = self.model.to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
    
    def choose_action(self, state):
        # Política epsilon-greedy con RNG determinista
        if self.rng.random() < self.epsilon:
            return self.rng.randrange(self.action_size)
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            with torch.no_grad():
                q_values = self.model(state_tensor)
            return q_values.argmax().item()
    
    def update(self, state, action, reward, next_state, next_action, done):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(device)
        
        # Valor Q actual
        q_value = self.model(state_tensor)[0, action]
        
        with torch.no_grad():
            if done:
                target = reward
            else:
                target = reward + self.gamma * self.model(next_state_tensor)[0, next_action]
        target = torch.tensor(target, device=device)
        
        loss = self.criterion(q_value, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def decay_epsilon(self):
        self.epsilon = max(self.epsilon * self.decay_rate, self.min_epsilon)

# -------------------------------------------------
# 5. Función de entrenamiento del agente SARSA
# -------------------------------------------------
def train_sarsa(env_wrapper, num_episodes=10000, seed=2024):
    # Establecer todas las semillas necesarias
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # Configurar el espacio de estados y acciones
    state_size = env_wrapper.observation_space.n
    action_size = env_wrapper.action_space.n
    
    # Inicializar el agente SARSA semigradiente
    agent = SarsaSemigrad(
        state_size=state_size,
        action_size=action_size,
        hidden_size=64,
        lr=0.001,
        gamma=0.99,
        epsilon=1.0,
        min_epsilon=0.1,
        decay_rate=0.995,
        seed=seed
    )
    
    episode_rewards = []      # Recompensa total por episodio
    episode_lengths = []      # Número de pasos por episodio
    training_errors = []      # Error de entrenamiento
    
    for episode in range(num_episodes):
        # Usar una semilla derivada para cada episodio
        episode_seed = seed + episode
        state = env_wrapper.reset(seed=episode_seed)
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
            loss = agent.update(state_one_hot, action, reward, next_state_one_hot, next_action, done)
            training_errors.append(loss)
            
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
    
    return episode_rewards, episode_lengths, training_errors, agent

# -------------------------------------------------
# 6. Clase para visualización de estadísticas
# -------------------------------------------------
class GraphVisualizer:
    def __init__(self, rewards, lengths, training_errors, rolling_length=500):
        self.rewards = rewards
        self.lengths = lengths
        self.training_errors = training_errors
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



