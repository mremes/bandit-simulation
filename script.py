import pylab
import numpy as np

from mab_simulator.agent import Agent
from mab_simulator.reward import NormallyDistributedRandomRewards
from mab_simulator.algorithms import EpsilonGreedyBandit, EpsilonFirstBandit, UCB1Bandit

from scipy import stats
from numpy import sqrt


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


rounds = 1000
n_actions = 5
rewards = NormallyDistributedRandomRewards(n_actions, mu_max=5, stddev=2, seed=42)
for i, ps in enumerate(rewards.params):
    mu, sigma = ps
    x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
    pylab.plot(x, stats.norm.pdf(x, mu, sigma), label=f'{i+1}')

pylab.legend(loc='upper left')
pylab.savefig(f'rewards_k{n_actions}_{rewards.mu_max}_{rewards.stddev}_{rewards.seed}')
pylab.show()

register_agent(Agent(rewards, EpsilonGreedyBandit(0.10)))
register_agent(Agent(rewards, EpsilonFirstBandit(0.10, rounds)))
register_agent(Agent(rewards, UCB1Bandit(sqrt(2))))

for _ in range(rounds):
    for m in agents:
        m.agent.play_round()
        m.record_metrics()

x = range(1, rounds+1)
for m in agents:
    y = m.pseudo_regret
    pylab.plot(x, y, label=m.agent.algorithm)

pylab.legend(loc='upper left')
pylab.savefig(f'regret_{rounds}')
pylab.show()

for m in agents:
    y = m.reward
    pylab.plot(x, y, label=m.agent.algorithm)

pylab.legend(loc='upper left')
pylab.savefig(f'reward_{rounds}')
pylab.show()
