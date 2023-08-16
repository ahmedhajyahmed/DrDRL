from abc import ABC, abstractmethod


class BaseEnv(ABC):

    def __init__(self, env_name, discrete_action_space=True):
        self.env_name = env_name
        self._env = self.create_env(discrete_action_space)
        self._obs_dim = self._env.observation_space.shape
        self._act_dim = self._env.action_space.n if discrete_action_space else self._env.action_space.shape[0]
        if not discrete_action_space:
            self.action_space_high = self._env.action_space.high
            self.action_space_low = self._env.action_space.low
            self._act_shape = self._env.action_space.shape
        self._turn = 0
        self._drifted_env = None
        self._is_drifted = False
        self._drifted_env_params = None
        self.discrete_action_space = discrete_action_space

    @abstractmethod
    def create_env(self, discrete_action_space):
        # return gym.make(self.env_name)
        raise NotImplementedError

    def reset(self):
        if self._is_drifted:
            observation = self._drifted_env.reset()
        else:
            observation = self._env.reset()
        return observation, self._turn

    def step(self, action):
        if self._is_drifted:
            observation, reward, done, info = self._drifted_env.step(action)
        else:
            observation, reward, done, info = self._env.step(action)
        self._turn += 1
        return observation, reward, done, self._turn, info

    def switch_to_drift(self):
        self._is_drifted = True

    def revert_drift(self):
        self._is_drifted = False

    @abstractmethod
    def drift(self, new_env_params):
        self.switch_to_drift()
        self._drifted_env_params = new_env_params

    def seed(self, seed=None):
        if self._is_drifted:
            self._drifted_env.seed(seed=seed)
        else:
            self._env.seed(seed=seed)

    def save(self, save_dir):
        raise NotImplementedError

    def load(self, load_dir):
        raise NotImplementedError

    def close(self):
        if self._is_drifted:
            self._drifted_env.close()
        else:
            self._env.close()

    def get_obs_dim(self):
        return self._obs_dim

    def get_act_dim(self):
        return self._act_dim

    def get_act_high_low(self):
        if not self.discrete_action_space:
            return self.action_space_high, self.action_space_low
        else:
            return None

    def get_act_shape(self):
        if not self.discrete_action_space:
            return self._act_shape
        else:
            return None

