from numpy.random import random
from .common import Bandit


class EpsilonGreedyBandit(Bandit):
    def __init__(self, epsilon):
        self.epsilon = epsilon

    def __str__(self):
        return f'e-greedy, e = {self.epsilon}'

    def draw(self, agent):
        sample = random()
        if sample < self.epsilon:
            return agent.any_action()
        else:
            return agent.get_best_action()
