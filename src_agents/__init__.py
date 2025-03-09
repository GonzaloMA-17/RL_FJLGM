from .agent import Agente

from .montecarloOnPolicy import MonteCarloOnPolicy
from .monteCarloOffPolicy import MonteCarloOffPolicy
from .sarsa import SARSA
from .qLearning import QLearning

# from .sarsaSemiGradiente import SarsaSemigradiente
from .sarsaSemiGradiente import SarsaSemigrad

from .politicas import epsilon_greedy_policy, pi_star_from_Q

__all__ = ['Agente',
           'MonteCarloOnPolicy', 
           'MonteCarloOffPolicy',
           'epsilon_greedy_policy',
           'SARSA',
           'QLearning',
           'pi_star_from_Q',
           'SarsaSemigrad',
           ]
