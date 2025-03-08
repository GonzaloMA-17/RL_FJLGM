import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

import numpy as np
import gymnasium as gym
from src_agents import Agente
from .politicas import epsilon_greedy_policy  # Usando la política epsilon-greedy
from tqdm import tqdm
import matplotlib.pyplot as plt
from typing import *
  # Tu clase base, ya definida en agent.py

# Función para convertir el estado (entero) a vector one-hot
def one_hot(state, state_size):
    vec = np.zeros(state_size, dtype=np.float32)
    vec[state] = 1.0
    return vec

# Definición de la red neuronal para SARSA semigradiente
class SarsaNet(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=64):
        super(SarsaNet, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, action_size)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Clase hija para implementar SARSA semigradiente
class SarsaSemigradiente(Agente):
    def __init__(self, env, epsilon=1.0, gamma=0.99, learning_rate=0.001, 
                 min_epsilon=0.1, decay_rate=0.995, hidden_size=64, seed=42):
        # Llamada al constructor de la clase base Agente
        # Observa que el constructor de Agente ya inicializa atributos como epsilon, gamma, Q, etc.
        # Aquí pasamos decay_rate como exploration_decay_rate
        super().__init__(env, epsilon, gamma, learning_rate, exploration_decay_rate=decay_rate, 
                         min_epsilon=min_epsilon, seed=seed)
        self.state_size = env.observation_space.n
        self.action_size = env.action_space.n
        self.hidden_size = hidden_size
        
        # Inicializar la red neuronal
        self.model = SarsaNet(self.state_size, self.action_size, hidden_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
    
    def seleccionar_accion(self, estado):
        """
        Sobrescribe el método abstracto para seleccionar una acción usando una política epsilon-greedy.
        El parámetro 'estado' es un entero, por lo que se convierte a vector one-hot.
        """
        estado_one_hot = one_hot(estado, self.state_size)
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)
        else:
            state_tensor = torch.FloatTensor(estado_one_hot).unsqueeze(0)
            q_values = self.model(state_tensor)
            return q_values.argmax().item()
    
    def entrenar(self, num_episodios):
        """
        Entrena al agente usando el algoritmo SARSA semigradiente.
        Devuelve dos listas: recompensas totales y longitud de cada episodio.
        """
        episode_rewards = []
        episode_lengths = []
        for episodio in range(num_episodios):
            state = self.env.reset()  # Aquí usamos el wrapper, que ya devuelve un entero
            state_one_hot = one_hot(state, self.state_size)
            action = self.seleccionar_accion(state)
            done = False
            total_reward = 0
            steps = 0
            
            while not done:
                next_state, reward, done, info = self.env.step(action)
                next_state_one_hot = one_hot(next_state, self.state_size)
                next_action = self.seleccionar_accion(next_state) if not done else None
                
                # Actualización en línea: SARSA semigradiente
                state_tensor = torch.FloatTensor(state_one_hot).unsqueeze(0)
                q_value = self.model(state_tensor)[0, action]
                with torch.no_grad():
                    if done:
                        target = reward
                    else:
                        next_state_tensor = torch.FloatTensor(next_state_one_hot).unsqueeze(0)
                        target = reward + self.gamma * self.model(next_state_tensor)[0, next_action]
                target = torch.tensor(target)
                loss = self.criterion(q_value, target)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                state_one_hot = next_state_one_hot
                action = next_action if not done else None
                total_reward += reward
                steps += 1
            
            self.decay_exploration()  # Método de la clase base Agente
            episode_rewards.append(total_reward)
            episode_lengths.append(steps)
            if (episodio+1) % 50 == 0:
                avg_reward = np.mean(episode_rewards[-50:])
                print(f"Episodio {episodio+1}, Recompensa Promedio: {avg_reward:.2f}, Epsilon: {self.epsilon:.2f}")
        
        return episode_rewards, episode_lengths
