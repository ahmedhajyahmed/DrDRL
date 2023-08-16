import copy
from dr_drl.envs.mountaincar.mountaincar_env_wrapper import MountainCarEnvWrapper
from dr_drl.envs.base_env import BaseEnv


class MountainCar(BaseEnv):

    def __init__(self, env_name, discrete_action_space=True):
        super().__init__(env_name, discrete_action_space)

    def create_env(self, discrete_action_space):
        return MountainCarEnvWrapper(discrete_action_space=discrete_action_space)

    def drift(self, new_env_params):
        super(MountainCar, self).drift(new_env_params)
        self._drifted_env = copy.deepcopy(self._env).unwrapped
        self._drifted_env.force = new_env_params['force']  # default is ?
        self._drifted_env.gravity = new_env_params['gravity']  # default is ?
        self._drifted_env.goal_velocity = new_env_params['goal_velocity']  # default is ?

    def save(self, save_dir):
        pass

    def load(self, load_dir):
        pass
