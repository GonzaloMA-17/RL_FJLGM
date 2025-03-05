from .agent import Agente
from .montecarloOnPolicy import MonteCarloOnPolicy
from .monteCarloOffPolicy import MonteCarloOffPolicy
from .sarsa import SARSA
from .qLearning import QLearning
from .politicas import epsilon_greedy_policy

__all__ = ['Agente',
           'MonteCarloOnPolicy', 
           'MonteCarloOffPolicy',
           'epsilon_greedy_policy',
           'SARSA',
           'QLearning']
