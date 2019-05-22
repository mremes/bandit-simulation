from numpy import random
from .common import Bandit
from ..reward import Reward


class EpsilonGreedyBandit(Bandit):
    def __init__(self, epsilon):
        self.epsilon = epsilon

    def __str__(self):
        return f'ε-greedy, ε = {self.epsilon}'

    def draw(self, agent) -> Reward:
        random.seed(agent.t)
        sample = random.random()
        if sample < self.epsilon:
            return agent.any_action()
        else:
            return agent.get_best_action()
