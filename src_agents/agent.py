import gymnasium as gym
from abc import ABC, abstractmethod
import numpy as np
import random

class Agente(ABC):
    def __init__(self, env: gym.Env, epsilon: float = 0.4, gamma: float = 1.0, learning_rate: float = 0.1,
                 exploration_decay_rate: float = 0.001, min_epsilon: float = 0.01, seed: int = 42):
        """
        Inicializa el agente para el aprendizaje por refuerzo, asegurando la reproducibilidad
        mediante el uso de una semilla común.
        
        Parámetros:..
        - env: El entorno de Gymnasium.
        - epsilon: La tasa de exploración inicial (probabilidad de exploración).
        - gamma: El factor de descuento (valor futuro de las recompensas).
        - learning_rate: La tasa de aprendizaje (cuánto ajustamos los valores de Q).
        - exploration_decay_rate: La tasa de decaimiento de epsilon.
        - min_epsilon: El valor mínimo de epsilon.
        - seed: La semilla para garantizar la reproducibilidad.
        """
        self.env = env
        self.epsilon = epsilon
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.exploration_decay_rate = exploration_decay_rate
        self.min_epsilon = min_epsilon
        self.estado = None
        self.accion = None
        self.seed = seed
        self.set_seed(seed)
        state_space_size = env.observation_space.n
        action_space_size = env.action_space.n
        self.Q = np.zeros([state_space_size, action_space_size])

    def set_seed(self, seed: int):
        np.random.seed(seed)
        random.seed(seed)
        self.env.reset(seed=seed)
        self.env.action_space.seed(seed)


    @abstractmethod
    def seleccionar_accion(self, estado):
        """Método abstracto para seleccionar una acción"""
        pass

    @abstractmethod
    def entrenar(self, num_episodios):
        """Método abstracto para entrenar al agente"""
        pass
    
    def decay_exploration(self):
        """Decae la tasa de exploración (epsilon) después de cada episodio"""
        self.epsilon = max(self.min_epsilon, self.epsilon * np.exp(-self.exploration_decay_rate))
    
    def actualizar_Q(self, obs, action, reward, next_obs):
        """
        Método común para actualizar la Q-table usando la fórmula general de Q-learning
        (aunque este será sobrescrito por los algoritmos específicos como SARSA, etc.)
        """
        best_next_action = np.argmax(self.Q[next_obs, :])  # Acción con el valor máximo en el siguiente estado
        td_target = reward + self.gamma * self.Q[next_obs, best_next_action]
        td_error = td_target - self.Q[obs, action]
        self.Q[obs, action] += self.learning_rate * td_error
