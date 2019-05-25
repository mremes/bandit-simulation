from numpy import random
from .common import Bandit
from ..reward import Reward
from numpy import sqrt, log, multiply, divide, add, inf


class EpsilonDecreasingBandit(Bandit):
    def __init__(self, epsilon):
        self.epsilon = epsilon

    def __str__(self):
        return f'ε-decreasing, ε = {self.epsilon}'

    def draw(self, agent) -> Reward:
        e_t = min(1, divide(self.epsilon, agent.t or self.epsilon))
        random.seed(agent.t)
        sample = random.random()
        if sample < e_t:
            return agent.any_action()
        else:
            return agent.get_best_action()
