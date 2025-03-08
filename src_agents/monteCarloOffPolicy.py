import numpy as np
from tqdm import tqdm
import gymnasium as gym
from src_agents import Agente
from .politicas import epsilon_greedy_policy

class MonteCarloOffPolicy(Agente):
    def __init__(self, 
                 env: gym.Env, 
                 epsilon: float = 0.8, 
                 gamma: float = 1.0, 
                 decay: bool = False, 
                 num_episodios: int = 5000):
        """
        Constructor del agente Monte Carlo Off-Policy con epsilon greedy.
        
        Args:
            env: Entorno de Gymnasium.
            epsilon: Parámetro de exploración inicial para la política de comportamiento (b).
            gamma: Factor de descuento.
            decay: Si True, se decae epsilon a lo largo de los episodios.
            num_episodios: Número total de episodios de entrenamiento.
        """
        super().__init__(env, epsilon, gamma)  # Llamada al constructor de la clase base Agente
        self.decay = decay
        self.num_episodios = num_episodios
        
        # Para contar visitas (usado en alpha = 1 / n_visits)
        self.n_visits = np.zeros([env.observation_space.n, env.action_space.n])
        
        # Variables para estadísticas
        self.stats = 0.0
        self.list_stats = [self.stats]
        self.episode_lengths = []

    def seleccionar_accion(self, estado):
        """
        Selecciona una acción usando la política de comportamiento (b),
        que aquí es epsilon-greedy con respecto a la Q actual.
        """
        return epsilon_greedy_policy(self.Q, self.epsilon, estado, self.env.action_space.n)

    def target_policy(self, state):
        """
        Define la política objetivo (π).
        En este ejemplo, es determinista (greedy con respecto a Q).
        """
        return np.argmax(self.Q[state])

    def behavior_policy_prob(self, state, action):
        """
        Probabilidad de tomar 'action' en 'state' bajo la política de comportamiento (b),
        asumiendo que b es epsilon-greedy respecto a Q.
        """
        nA = self.env.action_space.n
        probs = np.ones(nA, dtype=float) * (self.epsilon / nA)
        best_action = np.argmax(self.Q[state])
        probs[best_action] += (1.0 - self.epsilon)
        return probs[action]

    def target_policy_prob(self, state, action):
        """
        Probabilidad de tomar 'action' en 'state' bajo la política objetivo (π).
        Como π es determinista (greedy con respecto a Q), la acción óptima tiene
        probabilidad 1, y el resto 0.
        """
        best_action = self.target_policy(state)
        return 1.0 if action == best_action else 0.0

    def entrenar(self):
        """
        Entrena al agente mediante el algoritmo Monte Carlo Off-Policy
        usando Weighted Importance Sampling.
        """
        step_display = max(1, self.num_episodios // 10)
        
        for t in tqdm(range(self.num_episodios), desc="Entrenando Off-Policy (WIS)", unit="episodio", ncols=100):
            # Reiniciar el entorno
            state, info = self.env.reset(seed=1234)
            done = False
            episode = []
            episode_length = 0
            
            # Generar el episodio con la política de comportamiento (b)
            while not done:
                if self.decay:
                    # Reducir epsilon de forma gradual (ejemplo)
                    self.epsilon = min(1.0, 1000.0 / (t + 1))

                action = self.seleccionar_accion(state)
                next_state, reward, terminated, truncated, info = self.env.step(action)
                
                episode.append((state, action, reward))
                state = next_state
                done = terminated or truncated
                episode_length += 1

            self.episode_lengths.append(episode_length)

            # Retropropagación para calcular el retorno y actualizar Q con Weighted IS
            G = 0.0
            W = 1.0  # Factor de ponderación de importancia

            for (s, a, r) in reversed(episode):
                # Calcular el retorno acumulado
                G = r + self.gamma * G
                
                # Contar la visita (para promedio incremental)
                self.n_visits[s, a] += 1.0
                alpha = 1.0 / self.n_visits[s, a]
                
                # Actualización de Q con Weighted Importance Sampling
                self.Q[s, a] += alpha * W * (G - self.Q[s, a])

                # Si la política objetivo NO habría tomado esa acción (prob = 0), cortamos
                if self.target_policy_prob(s, a) == 0.0:
                    break

                # Actualizar el factor de importancia
                # W = W * (π(a|s) / b(a|s))
                W *= (self.target_policy_prob(s, a) / self.behavior_policy_prob(s, a))

            # Actualizar estadísticas
            self.stats += G
            self.list_stats.append(self.stats / (t + 1))
            
            # Mostrar estadísticas de progreso
            if t % step_display == 0 and t != 0:
                print(f"Episodio {t}, recompensa promedio: {self.list_stats[-1]:.2f}, epsilon: {self.epsilon:.2f}")

        return self.Q, self.list_stats, self.episode_lengths
