import numpy as np
import gymnasium as gym
from src_agents import Agente
from .politicas import epsilon_greedy_policy  # Asegúrate de tener esta política implementada
from tqdm import tqdm

class SARSA(Agente):
    def __init__(self, env: gym.Env, epsilon: float = 0.9, alpha: float = 0.1, gamma: float = 1.0, 
                 decay: bool = False, num_episodios: int = 5000):
        """
        Inicializa el agente SARSA.
        - env: Entorno de Gymnasium.
        - epsilon: Tasa de exploración inicial.
        - alpha: Tasa de aprendizaje.
        - gamma: Factor de descuento.
        - decay: Si True, epsilon decae con el tiempo.
        - num_episodios: Número total de episodios a entrenar.
        """
        super().__init__(env, epsilon, gamma)  
        self.alpha = alpha
        self.decay = decay  
        self.num_episodios = num_episodios
        self.stats = 0.0
        self.list_stats = []
        self.episode_lengths = []  

    def seleccionar_accion(self, estado):
        """
        Selecciona una acción utilizando la política epsilon-greedy.
        """
        return epsilon_greedy_policy(self.Q, self.epsilon, estado, self.env.action_space.n)

    def entrenar(self):
        """
        Entrena al agente utilizando el algoritmo SARSA.
        """
        for episodio in tqdm(range(self.num_episodios), desc="Entrenando", unit="episodio", ncols=100):
            state, info = self.env.reset(seed=1234)
            action = self.seleccionar_accion(state)
            done = False
            episode_length = 0  
            total_reward = 0  

            while not done:
                new_state, reward, terminated, truncated, info = self.env.step(action)
                new_action = self.seleccionar_accion(new_state)

                # Actualización de la función de acción-valor Q(s, a)
                self.Q[state, action] += self.alpha * (reward + self.gamma * self.Q[new_state, new_action] - self.Q[state, action])
                
                # Actualizar estado y acción
                state = new_state
                action = new_action
                done = terminated or truncated
                episode_length += 1  
                total_reward += reward  

                # Decaimiento de epsilon si está activado
                if self.decay:
                    self.epsilon = max(0.01, self.epsilon * 0.995)

            # Registrar estadísticas
            self.episode_lengths.append(episode_length)
            self.list_stats.append(total_reward)

            # Imprimir estado del entrenamiento en los episodios clave
            if episodio in [2500, 3000, 3500, 4000, 4500]:
                exito_promedio = np.mean(self.list_stats)
                print(f"\nEpisodio {episodio}, éxito promedio: {exito_promedio:.2f}, epsilon: {self.epsilon:.2f}")

        # Mostrar la tabla Q final después del entrenamiento
        print("\nValor Q final con SARSA:")
        print(self.Q)

        return self.Q, self.list_stats, self.episode_lengths
