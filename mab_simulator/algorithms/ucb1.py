from numpy import sqrt, log, multiply, divide, add, inf
from .common import Bandit
from ..reward import Reward


class UCB1Bandit(Bandit):
    def __init__(self, c):
        self.c = c

    def __str__(self):
        return f'ucb1, c = {self.c}'

    @staticmethod
    def _get_uncertainty(agent, action):
        n = agent.N[action.label]
        if not n:
            return inf
        return sqrt(divide(multiply(2, log(agent.t)), (agent.N[action.label])))

    @staticmethod
    def _get_action_value(agent, action):
        return agent.Q[action.label]

    def draw(self, agent) -> Reward:
        res_action: Reward = None
        res_value = 0
        for a in agent.actions.rewards.values():
            q = self._get_action_value(agent, a)
            p = self._get_uncertainty(agent, a)
            value = add(p, q)
            if value > res_value:
                res_action = a
                res_value = value
        return res_action
