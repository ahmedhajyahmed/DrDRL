from abc import ABC, abstractmethod
import tensorflow as tf
import numpy as np
import random


class Runner(ABC):

    def __init__(self, env, agent, train_episodes, test_episodes, max_steps_per_episode, reward_threshold=None,
                 validation_period=10, tracker=None, seed=42):
        self._env = env
        self._agent = agent
        self._train_episodes = train_episodes
        self._test_episodes = test_episodes
        self._run_training = True
        self._run_testing = True
        self._max_steps_per_episode = max_steps_per_episode
        self._reward_threshold = reward_threshold
        self._validation_period = validation_period
        self._tracker = tracker  # needs to be changed a little
        self.set_seed(seed)

    def set_seed(self, seed):
        self._env.seed(seed)  # Set the seed to keep the environment consistent across runs
        tf.random.set_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

    @abstractmethod
    def run_one_step(self, state, episode_metrics):
        raise NotImplementedError

    @abstractmethod
    def run_episode(self, episodes):
        raise NotImplementedError

    @abstractmethod
    def run_training(self, instance_name, save_dir=None):
        raise NotImplementedError

    @abstractmethod
    def run_testing(self):
        raise NotImplementedError

    @property
    def env(self):
        return self._env
