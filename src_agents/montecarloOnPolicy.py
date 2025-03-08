import numpy as np
import gymnasium as gym
from tqdm import tqdm
from typing import *
from src_agents import Agente
from .politicas import epsilon_greedy_policy

class MonteCarloOnPolicy(Agente):
    def __init__(self, 
                 env: gym.Env, 
                 epsilon: float = 0.4, 
                 gamma: float = 1.0, 
                 decay: bool = False, 
                 num_episodios: int = 5000):
        """
        Inicializa el agente Monte Carlo On-Policy.
        
        Parámetros:
        - env: Entorno de Gymnasium.
        - epsilon: Tasa de exploración inicial.
        - gamma: Factor de descuento.
        - decay: Indica si se decae la exploración con cada episodio.
        - num_episodios: Número total de episodios de entrenamiento.
        """
        # Llamamos al constructor de la clase base (Agente)
        super().__init__(env, epsilon, gamma)
        
        self.decay = decay
        self.num_episodios = num_episodios
        
        # Contador de visitas para cada (estado, acción)
        self.n_visits = np.zeros([env.observation_space.n, env.action_space.n])
        
        # Para almacenar estadísticas de entrenamiento
        self.stats = 0.0
        self.list_stats = []
        self.episode_lengths = []
    
    def seleccionar_accion(self, estado: int) -> int:
        """
        Selecciona una acción usando la política epsilon-greedy definida en 'politicas.py'.
        """
        return epsilon_greedy_policy(self.Q, self.epsilon, estado, self.env.action_space.n)

    def entrenar(self) -> Tuple[np.ndarray, List[float], List[int]]:
        """
        Entrena al agente utilizando Monte Carlo On-Policy sin límite de pasos por episodio.
        
        Devuelve:
        - self.Q: La Q-table entrenada (array 2D de tamaño [n_estados, n_acciones]).
        - self.list_stats: Lista con la recompensa media acumulada tras cada episodio.
        - self.episode_lengths: Lista con la longitud de cada episodio.
        """
        for t in tqdm(range(self.num_episodios), desc="Entrenando", unit="episodio", ncols=100):
            state, _ = self.env.reset()
            done = False
            episode = []
            episode_length = 0
            
            while not done:
                # Decaimiento opcional de epsilon
                if self.decay:
                    # Evitar que epsilon supere 1.0
                    self.epsilon = max(0.01, min(1.0, 1000.0 / (t + 1)))
                
                # Seleccionar acción
                action = self.seleccionar_accion(state)
                
                # Interactuar con el entorno
                new_state, reward, terminated, truncated, info = self.env.step(action)
                
                # Almacenar la transición
                episode.append((state, action, reward))
                
                state = new_state
                done = terminated or truncated
                episode_length += 1
            
            # Guardar la longitud del episodio
            self.episode_lengths.append(episode_length)
            
            # Calcular retorno acumulado y actualizar Q de atrás hacia delante
            G = 0.0
            for (s, a, r) in reversed(episode):
                G = r + self.gamma * G
                self.n_visits[s, a] += 1.0
                alpha = 1.0 / self.n_visits[s, a]  # Promedio incremental
                self.Q[s, a] += alpha * (G - self.Q[s, a])
            
            # Actualizar estadísticas
            self.stats += G
            self.list_stats.append(self.stats / (t + 1))
        
        return self.Q, self.list_stats, self.episode_lengths
