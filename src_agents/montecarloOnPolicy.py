import numpy as np
import gymnasium as gym
from src_agents import Agente
from .politicas import epsilon_greedy_policy  # Asumiendo que tienes esta política
from tqdm import tqdm
import matplotlib.pyplot as plt

class MonteCarloOnPolicy(Agente):
    def __init__(self, env: gym.Env, epsilon: float = 0.4, gamma: float = 1.0, decay: bool = False, num_episodios: int = 5000):
        super().__init__(env, epsilon, gamma)  # Llamamos al constructor de la clase base
        self.decay = decay  # Aseguramos que decay se asigne correctamente
        self.num_episodios = num_episodios
        self.n_visits = np.zeros([env.observation_space.n, env.action_space.n])  # Para contar visitas
        self.stats = 0.0
        self.list_stats = [self.stats]
        self.episode_lengths = []  # Lista para almacenar las longitudes de los episodios

    def seleccionar_accion(self, estado):
        """Selecciona una acción utilizando la política epsilon-greedy"""
        return epsilon_greedy_policy(self.Q, self.epsilon, estado, self.env.action_space.n)

    def entrenar(self):
        """Entrena al agente utilizando el algoritmo Monte Carlo On-Policy"""
        step_display = self.num_episodios // 10
        
        # Usamos tqdm para mostrar la barra de progreso
        for t in tqdm(range(self.num_episodios), desc="Entrenando", unit="episodio", ncols=100):
            state, info = self.env.reset(seed=1234)  # Reiniciar el entorno con la semilla
            done = False
            episode = []  # Almacena tuplas (estado, acción, recompensa)
            episode_length = 0  # Inicializamos la longitud del episodio
            
            while not done:
                if self.decay:
                    # Reducir epsilon de forma gradual
                    self.epsilon = min(1.0, 1000.0 / (t + 1))
                action = self.seleccionar_accion(state)
                new_state, reward, terminated, truncated, info = self.env.step(action)
                episode.append((state, action, reward))  # Guardar la transición
                state = new_state
                done = terminated or truncated
                episode_length += 1  # Contabilizamos la longitud del episodio

            # Almacenar la longitud del episodio
            self.episode_lengths.append(episode_length)

            # Calcular retornos acumulados y actualizar Q
            G = 0.0
            for state, action, reward in reversed(episode):
                G = reward + self.gamma * G  # Retorno acumulado
                self.n_visits[state, action] += 1.0
                alpha = 1.0 / self.n_visits[state, action]  # Factor de promedio incremental
                self.Q[state, action] += alpha * (G - self.Q[state, action])  # Actualización Q

            # Actualizar estadísticas
            self.stats += G
            self.list_stats.append(self.stats / (t + 1))
            
            # Mostrar estadísticas
            if t % step_display == 0 and t != 0:
                print(f"Episodio {t}, éxito promedio: {self.stats / (t + 1):.2f}, epsilon: {self.epsilon:.2f}")

        return self.Q, self.list_stats, self.episode_lengths