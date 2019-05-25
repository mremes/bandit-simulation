import os
import pylab
import numpy as np

from bandit.agent import Agent
from bandit.reward import NormallyDistributedRandomRewards
from bandit.algorithms import EpsilonGreedyBandit, EpsilonFirstBandit, EpsilonDecreasingBandit, UCB1Bandit

from scipy import stats
from numpy import sqrt
from itertools import accumulate


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


def save(name, *args, **kwargs):
    try:
        # running as script
        __file__
        path = f'plots/{name}'
    except NameError:
        # running in repl from root
        path = f'notebooks/plots/{name}'
    finally:
        pylab.savefig(path, *args, **kwargs)


rounds = 1000
n_actions = 5
rewards = NormallyDistributedRandomRewards(n_actions, mu_max=5, stddev=2, seed=42)
for i, ps in enumerate(rewards.params):
    mu, sigma = ps
    x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
    pylab.plot(x, stats.norm.pdf(x, mu, sigma), label=f'{i+1}')

pylab.legend(loc='upper left')
save('rewards')
pylab.show()

register_agent(Agent(rewards, EpsilonGreedyBandit(0.08)))
register_agent(Agent(rewards, EpsilonFirstBandit(0.03, rounds)))
register_agent(Agent(rewards, EpsilonDecreasingBandit(3)))
register_agent(Agent(rewards, UCB1Bandit(sqrt(2))))

for _ in range(rounds):
    for m in agents:
        m.agent.play_round()
        m.record_metrics()

x = range(1, rounds+1)

optimal_regret = [0] * rounds
pylab.plot(x, optimal_regret, label='optimal', linestyle='--')

for m in agents:
    y = m.pseudo_regret
    pylab.plot(x, y, label=m.agent.algorithm)

pylab.legend(loc='upper left')
pylab.xlabel('t')
pylab.ylabel('katumus')
save(f'regret_{rounds}')
pylab.show()

optimal_reward = list(accumulate([rewards.best_action().expected] * rounds))
pylab.plot(x, optimal_reward, label='optimal', linestyle='--')

for m in agents:
    y = m.reward
    pylab.plot(x, y, label=m.agent.algorithm)

pylab.xlabel('t')
pylab.ylabel('tuotto')
pylab.legend(loc='upper left')
save(f'reward_{rounds}')
pylab.show()
