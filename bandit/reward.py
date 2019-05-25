import numpy
from .common import container_for
from typing import Dict


class Reward:

    def __init__(self, label, distribution: numpy.random, expected: float, args: tuple):
        self.label = label
        self.expected = expected
        self.distribution_fn = lambda: distribution(*args)

    def sample(self):
        return self.distribution_fn()

    def __str__(self):
        return self.label


class Rewards:

    def __init__(self, k):
        self.k = k
        self.rewards: Dict[str, Reward] = container_for(k, None)
        self.best: str = None
        self.init_rewards()

    def __len__(self):
        return self.k

    def __getitem__(self, item):
        if isinstance(item, Reward):
            return self.rewards[item.label]
        return self.rewards[item]

    def get_container(self, default_value=0):
        return dict.fromkeys(self.rewards.keys(), default_value)

    def best_action(self):
        return self.rewards[self.best]

    def any_action(self, seed=0):
        numpy.random.seed(seed)
        idx = numpy.random.randint(len(self.rewards))
        label = sorted(list(self.rewards.keys()))[idx]
        return self.rewards[label]

    def init_rewards(self):
        raise NotImplementedError


class NormallyDistributedRandomRewards(Rewards):

    def __init__(self, k, mu_max=10, stddev=0.5, seed=10):
        self.mu_max = mu_max
        self.stddev = stddev
        self.seed = seed
        self.params = []
        super().__init__(k)

    def init_rewards(self):
        best = 0
        numpy.random.seed(self.seed)
        for reward, _ in self.rewards.items():
            loc = numpy.random.random() * self.mu_max
            if loc > best:
                self.best = reward
                best = loc
            params = (loc, self.stddev)
            self.params.append(params)
            self.rewards[reward] = Reward(reward, numpy.random.normal, loc, params)
