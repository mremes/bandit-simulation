import numpy
import random
from .common import container_for
from typing import Dict


class Reward:

    def __init__(self, label, distribution: numpy.random, expected: float, args: tuple):
        self.label = label
        self.expected = expected
        self.distribution_fn = lambda: distribution(*args)

    def sample(self):
        return self.distribution_fn()


class Rewards:

    def __init__(self, k):
        self.k = k
        self.rewards: Dict[str, Reward] = container_for(k, None)
        self.best: str = None
        self.init_rewards()

    def __len__(self):
        return self.k

    def __getitem__(self, item):
        return self.rewards[item]

    def get_container(self, default_value=0):
        return dict.fromkeys(self.rewards.keys(), default_value)

    def best_action(self):
        return self.rewards[self.best]

    def any_action(self):
        return self.rewards[random.choice(list(self.rewards.keys()))]

    def init_rewards(self):
        raise NotImplementedError


class NormallyDistributedRandomRewards(Rewards):

    def __init__(self, k, stddev=0.5):
        self.stddev = stddev
        super().__init__(k)

    def init_rewards(self):
        best = 0
        for reward, _ in self.rewards.items():
            loc = numpy.random.randint(1, self.k+1)
            if loc > best:
                self.best = reward
                best = loc
            params = (loc, self.stddev)
            self.rewards[reward] = Reward(reward, numpy.random.normal, loc, params)
