import operator

from .reward import Rewards, Reward
from .algorithms import Bandit


class Agent:
    def __init__(self, rewards: Rewards, algorithm: Bandit):
        self.actions = rewards
        self.algorithm = algorithm
        self.Q = rewards.get_container()
        self.N = rewards.get_container()
        self.best_action = None
        self.t = 0
        self.reward = 0
        self.pseudo_regret = 0
        self.regret = 0

    def play_round(self):
        reward: Reward = self.algorithm.draw(self)
        label = reward.label
        win = reward.sample()
        regret = self.actions.best_action().expected - win
        expected_regret = self.actions.best_action().expected - reward.expected

        self.N[label] += 1
        self.Q[label] += (1 / self.N[label]) * (win - self.Q[label])
        self.best_action = self.actions[max(self.Q.items(), key=operator.itemgetter(1))[0]]

        self.t += 1
        self.reward += win
        self.regret += regret
        self.pseudo_regret += expected_regret

    def any_action(self):
        return self.actions.any_action()

    def get_best_action(self):
        return self.best_action or self.any_action()
