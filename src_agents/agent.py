import gymnasium as gym
from abc import ABC, abstractmethod
import numpy as np

class Agente(ABC):
    def __init__(self, env: gym.Env, epsilon: float = 0.4, gamma: float = 1.0):
        self.env = env
        self.epsilon = epsilon
        self.gamma = gamma
        self.estado = None
        self.accion = None
        self.Q = np.zeros([env.observation_space.n, env.action_space.n])  # Inicializamos la matriz Q

    @abstractmethod
    def seleccionar_accion(self, estado):
        """Método abstracto para seleccionar una acción"""
        pass

    @abstractmethod
    def entrenar(self, num_episodios):
        """Método abstracto para entrenar al agente"""
        pass
