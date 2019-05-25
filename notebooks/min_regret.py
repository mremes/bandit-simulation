from typing import List

import pylab

from bandit.agent import Agent
from bandit.reward import NormallyDistributedRandomRewards
from bandit.algorithms import EpsilonGreedyBandit


class AgentMetrics:
    def __init__(self, agent: Agent):
        self.agent = agent
        self.reward = []
        self.regret = []
        self.pseudo_regret = []

    def record_metrics(self):
        self.reward.append(self.agent.reward)
        self.regret.append(self.agent.regret)
        self.pseudo_regret.append(self.agent.pseudo_regret)


agents: List[AgentMetrics] = []


def register_agent(a: Agent):
    agents.append(AgentMetrics(a))


rounds = 1000
n_actions = 5
rewards = NormallyDistributedRandomRewards(n_actions, mu_max=5, stddev=2, seed=42)

for i in range(1, 100):
    register_agent(Agent(rewards, EpsilonGreedyBandit(i/100)))

for _ in range(rounds):
    for m in agents:
        m.agent.play_round()
        m.record_metrics()

x = list(map(lambda x: x/100, range(1, 100)))
y = [a.pseudo_regret[-1] for a in agents]
pylab.plot(x, y)
pylab.xticks(list(map(lambda x: (x/100)*10, range(11))))
pylab.show()
