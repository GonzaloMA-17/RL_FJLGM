import numpy as np
import gymnasium as gym
from src_agents import Agente
from .politicas import epsilon_greedy_policy, random_epsilon_greedy_policy
from tqdm import tqdm

class MonteCarloOffPolicy(Agente):
    def __init__(self, env: gym.Env, epsilon: float = 0.9, gamma: float = 1.0, 
                 decay: bool = True, min_epsilon: float = 0.05, epsilon_decay_rate: float = 0.999, 
                 num_episodios: int = 5000):
        """
        Inicializa el agente Monte Carlo Off-Policy.
        - env: Entorno de Gymnasium.
        - epsilon: Tasa inicial de exploración.
        - gamma: Factor de descuento.
        - decay: Si True, epsilon decae con el tiempo.
        - min_epsilon: Límite mínimo para epsilon.
        - epsilon_decay_rate: Factor de reducción de epsilon por episodio.
        - num_episodios: Número de episodios a entrenar.
        """
        super().__init__(env, epsilon, gamma)  
        self.decay = decay  
        self.min_epsilon = min_epsilon
        self.epsilon_decay_rate = epsilon_decay_rate
        self.num_episodios = num_episodios
        
        self.Q = np.zeros((env.observation_space.n, env.action_space.n))
        self.C = np.zeros((env.observation_space.n, env.action_space.n))  # Acumulador de pesos
        self.stats = []
        self.episode_lengths = []
        self.successes = []

    def seleccionar_accion(self, estado):
        """Selecciona una acción utilizando la política epsilon-greedy."""
        return epsilon_greedy_policy(self.Q, self.epsilon, estado, self.env.action_space.n)

    def entrenar(self):
        """
        Entrena al agente utilizando el algoritmo Monte Carlo Off-Policy con 
        importancia ponderada.
        """
        for episodio in tqdm(range(self.num_episodios), desc="Entrenando", unit="episodio", ncols=100):
            state, _ = self.env.reset()
            episodio_experiencia = []  
            done = False
            step_count = 0  
            total_reward = 0  

            # **Generación del episodio siguiendo la política de comportamiento (b)**
            while not done:
                action = self.seleccionar_accion(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                # Guardar la transición del episodio
                episodio_experiencia.append((state, action, reward))
                state = next_state
                step_count += 1
                total_reward += reward  

            # **Actualización Off-Policy de Q-values usando importancia ponderada**
            G = 0.0  
            W = 1.0  

            for (state_t, action_t, reward_t) in reversed(episodio_experiencia):
                G = self.gamma * G + reward_t
                self.C[state_t, action_t] += W  
                self.Q[state_t, action_t] += (W / self.C[state_t, action_t]) * (G - self.Q[state_t, action_t])
                
                # Si la acción tomada no es la acción óptima según la política target, detenemos la actualización
                if action_t != np.argmax(self.Q[state_t]):
                    break
                
                # Actualización del peso de importancia
                behavior_prob = random_epsilon_greedy_policy(self.Q, self.epsilon, state_t, self.env.action_space.n)[action_t]
                W /= behavior_prob  

            # **Decay de epsilon**
            if self.decay:
                self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay_rate)

            # **Registro de estadísticas**
            self.episode_lengths.append(step_count)
            self.stats.append(total_reward)
            self.successes.append(1 if total_reward > 0 else 0)

            # **Mostrar estadísticas en episodios clave**
            if episodio in [2500, 3000, 3500, 4000, 4500]:
                exito_promedio = np.mean(self.successes)
                print(f"\nEpisodio {episodio}, éxito promedio: {exito_promedio:.2f}, epsilon: {self.epsilon:.2f}")

        # **Mostrar la tabla Q final**
        print("\nValor Q final con MonteCarloOffPolicy:")
        print(self.Q)

        return self.Q, self.stats, self.episode_lengths
