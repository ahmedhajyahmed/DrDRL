from abc import ABC, abstractmethod


class Agent(ABC):

    def __init__(self, obs_dim, act_dim):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self._training = True

    @abstractmethod
    def action(self, state, pruning=False):
        raise NotImplementedError

    @abstractmethod
    def update(self, data, pruning=False):
        raise NotImplementedError

    def switch_train(self):
        self._training = True

    def switch_test(self):
        self._training = False

    @abstractmethod
    def save(self, dir_path):
        raise NotImplementedError

    @abstractmethod
    def load(self, dir_path):
        raise NotImplementedError

    @abstractmethod
    def setup_pruning(self):
        raise NotImplementedError

