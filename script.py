import pylab

from mab_simulator.agent import Agent
from mab_simulator.reward import NormallyDistributedRandomRewards
from mab_simulator.algorithms import EpsilonGreedyBandit, EpsilonFirstBandit


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


agents = []


def register_agent(a: Agent):
    agents.append(AgentMetrics(a))


rounds = 10000
rewards = NormallyDistributedRandomRewards(10, 1)

register_agent(Agent(rewards, EpsilonGreedyBandit(0.1)))
register_agent(Agent(rewards, EpsilonFirstBandit(0.1, rounds)))

for _ in range(rounds):
    for m in agents:
        m.agent.play_round()
        m.record_metrics()

x = range(1, rounds+1)
for m in agents:
    y = m.pseudo_regret
    pylab.plot(x, y, label=m.agent.name)

pylab.legend(loc='upper left')
pylab.show()
