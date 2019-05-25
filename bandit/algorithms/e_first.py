from .common import Bandit
from ..reward import Reward


class EpsilonFirstBandit(Bandit):
    def __init__(self, epsilon, T):
        self.epsilon = epsilon
        self.T = T

    def __str__(self):
        return f'ε-first, ε = {self.epsilon}'

    def draw(self, agent) -> Reward:
        if agent.t < self.epsilon * self.T:
            return agent.any_action()
        else:
            return agent.get_best_action()
