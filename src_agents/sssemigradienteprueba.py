import numpy as np
import gymnasium as gym
from tqdm import tqdm
from typing import *
import torch
import torch.nn as nn
import torch.optim as optim
from src_agents import Agente
from .politicas import epsilon_greedy_policy

class SarsaSemigradiente(Agente):
    def __init__(self, 
                 env: gym.Env, 
                 epsilon: float = 0.4, 
                 gamma: float = 0.99, 
                 alpha: float = 0.1,
                 decay: bool = False, 
                 num_episodios: int = 5000,
                 hidden_size: int = 128):
        """
        Inicializa el agente Sarsa Semigradiente.
        
        Parámetros:
        - env: Entorno de Gymnasium.
        - epsilon: Tasa de exploración inicial.
        - gamma: Factor de descuento.
        - alpha: Tasa de aprendizaje para el optimizador.
        - decay: Indica si se decae la exploración con cada episodio.
        - num_episodios: Número total de episodios de entrenamiento.
        - hidden_size: Tamaño de la capa oculta en la red neuronal.
        """
        # Llamamos al constructor de la clase base (Agente)
        super().__init__(env, epsilon, gamma)
        
        self.decay = decay
        self.num_episodios = num_episodios
        self.alpha = alpha
        
        # Definir la aproximación de la función Q con una red neuronal
        self.n_inputs = env.observation_space.n  # Tamaño del espacio de estados
        self.n_outputs = env.action_space.n      # Tamaño del espacio de acciones
        
        # Arquitectura de la red
        self.model = nn.Sequential(
            nn.Linear(self.n_inputs, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, self.n_outputs)
        )
        
        # Optimizador
        self.optimizer = optim.SGD(self.model.parameters(), lr=alpha)
        
        # Para almacenar estadísticas de entrenamiento
        self.stats = 0.0
        self.list_stats = []
        self.episode_lengths = []
    
    def state_to_tensor(self, state: int) -> torch.Tensor:
        """
        Convierte un índice de estado a un tensor one-hot.
        """
        state_tensor = torch.zeros(self.n_inputs)
        state_tensor[state] = 1.0
        return state_tensor.unsqueeze(0)  # Añadir dimensión de batch
    
    def get_q_values(self, state: int) -> torch.Tensor:
        """
        Obtiene los valores Q para un estado dado usando la red neuronal.
        """
        state_tensor = self.state_to_tensor(state)
        return self.model(state_tensor)
    
    def seleccionar_accion(self, estado: int) -> int:
        """
        Selecciona una acción usando política epsilon-greedy.
        """
        with torch.no_grad():
            q_values = self.get_q_values(estado).squeeze(0).numpy()
        
        # Usar política epsilon-greedy con los valores Q aproximados
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()  # Exploración
        else:
            return np.argmax(q_values)  # Explotación
    
    def update_q(self, state: int, action: int, reward: float, next_state: int, next_action: int) -> float:
        """
        Actualiza la aproximación de la función Q usando Sarsa Semigradiente.
        
        Devuelve:
        - loss: El error de la actualización.
        """
        # Calcular el valor Q actual
        q_values = self.get_q_values(state)
        current_q = q_values[0, action]
        
        # Calcular el valor Q objetivo
        with torch.no_grad():
            next_q_values = self.get_q_values(next_state)
            next_q = next_q_values[0, next_action]
            target_q = reward + self.gamma * next_q
        
        # Calcular la pérdida
        loss = 0.5 * (target_q - current_q) ** 2
        
        # Optimización
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

    def entrenar(self) -> Tuple[nn.Module, List[float], List[int]]:
        """
        Entrena al agente utilizando Sarsa Semigradiente.
        
        Devuelve:
        - self.model: La red neuronal entrenada.
        - self.list_stats: Lista con la recompensa media acumulada tras cada episodio.
        - self.episode_lengths: Lista con la longitud de cada episodio.
        """
        for t in tqdm(range(self.num_episodios), desc="Entrenando", unit="episodio", ncols=100):
            state, _ = self.env.reset()
            action = self.seleccionar_accion(state)
            done = False
            episode_reward = 0.0
            episode_length = 0
            
            while not done:
                # Decaimiento opcional de epsilon
                if self.decay:
                    self.epsilon = max(0.01, min(1.0, 1000.0 / (t + 1)))
                
                # Interactuar con el entorno
                next_state, reward, terminated, truncated, info = self.env.step(action)
                
                # Seleccionar la siguiente acción
                next_action = self.seleccionar_accion(next_state)
                
                # Actualizar la función Q
                loss = self.update_q(state, action, reward, next_state, next_action)
                
                state = next_state
                action = next_action
                done = terminated or truncated
                episode_reward += reward
                episode_length += 1
            
            # Guardar la longitud del episodio
            self.episode_lengths.append(episode_length)
            
            # Actualizar estadísticas
            self.stats += episode_reward
            self.list_stats.append(self.stats / (t + 1))
        
        return self.model, self.list_stats, self.episode_lengths
    
    def guardar_modelo(self, ruta: str) -> None:
        """
        Guarda el modelo en la ruta especificada.
        """
        torch.save(self.model.state_dict(), ruta)
    
    def cargar_modelo(self, ruta: str) -> None:
        """
        Carga el modelo desde la ruta especificada.
        """
        self.model.load_state_dict(torch.load(ruta))