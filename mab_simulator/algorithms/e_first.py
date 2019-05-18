from .common import Bandit


class EpsilonFirstBandit(Bandit):
    def __init__(self, epsilon, T):
        self.epsilon = epsilon
        self.T = T

    def __str__(self):
        return f'e-first, e = {self.epsilon}'

    def draw(self, agent):
        if agent.t < self.epsilon * self.T:
            return agent.any_action()
        else:
            return agent.get_best_action()