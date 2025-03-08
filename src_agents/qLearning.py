from tqdm import tqdm
import matplotlib.pyplot as plt
from typing import *
import numpy as np
import gymnasium as gym
from src_agents import Agente
from .politicas import epsilon_greedy_policy  

class QLearning(Agente):
    def __init__(self, env: gym.Env, hiperparametros: Dict[str, Any]):
        """
        Inicializa el agente Q-learning.

        Parámetros:
        - env: El entorno de Gymnasium.
        - hiperparametros: Diccionario con parámetros como tasa de aprendizaje,
            factor de descuento, tasa de exploración, etc.
        """
        # Pasamos los parámetros correctos desde hiperparametros a la clase base
        super().__init__(env, 
                         epsilon=hiperparametros.get("exploration_rate", 0.4),
                         gamma=hiperparametros.get("discount_rate", 0.99),
                         learning_rate=hiperparametros.get("learning_rate", 0.1),
                         exploration_decay_rate=hiperparametros.get("exploration_decay_rate", 0.001),
                         min_epsilon=hiperparametros.get("min_exploration_rate", 0.01),
                         seed=hiperparametros.get("seed", 42))  # Asegurarse de pasar la semilla

    def seleccionar_accion(self, estado):
        """
        Implementa la política epsilon-greedy para Q-learning:
        - Con probabilidad epsilon, se elige una acción aleatoria (exploración).
        - Con probabilidad 1 - epsilon, se elige la acción con el valor máximo Q (explotación).
        """
        return epsilon_greedy_policy(self.Q, self.epsilon, estado, self.env.action_space.n)
    
    def entrenar(self, num_episodios: int):
        """
        Entrena al agente utilizando el algoritmo Q-learning.
        
        Devuelve:
        - self.Q: La Q-table entrenada.
        - self.stats: Las estadísticas de recompensa acumulada por episodio.
        - self.episode_lengths: Las longitudes de los episodios.
        """
        self.stats = 0.0  # Inicializar estadísticas
        self.list_stats = []  # Lista para almacenar estadísticas de cada episodio
        self.episode_lengths = []  # Lista para almacenar las longitudes de cada episodio
        
        for episodio in tqdm(range(num_episodios), desc="Entrenando", unit="episodio", ncols=100):
            estado, info = self.env.reset()
            done = False
            episode_length = 0  # Inicializar la longitud del episodio
            while not done:
                # Seleccionar la acción usando la política epsilon-greedy
                accion = self.seleccionar_accion(estado)
                
                # Interactuar con el entorno y obtener la siguiente transición
                siguiente_estado, recompensa, terminado, truncado, _ = self.env.step(accion)
                
                # Actualizar la Q-table con la recompensa y el siguiente estado
                self.actualizar_Q(estado, accion, recompensa, siguiente_estado)
                
                # Actualizar el estado
                estado = siguiente_estado
                done = terminado or truncado
                episode_length += 1  # Contabilizar la longitud del episodio

            # Almacenar la longitud del episodio
            self.episode_lengths.append(episode_length)

            # Actualizar las estadísticas
            self.stats += recompensa
            self.list_stats.append(self.stats / (episodio + 1))
            
            # Decrecer epsilon al final de cada episodio
            self.decay_exploration()

        # Devolver la Q-table entrenada, estadísticas y longitudes de episodios
        return self.Q, self.list_stats, self.episode_lengths
