import copy
from dr_drl.envs.acrobot.acrobot_env_wrapper import AcrobotEnvWrapper
from dr_drl.envs.base_env import BaseEnv


class Acrobot(BaseEnv):

    def __init__(self, env_name, discrete_action_space=True):
        super().__init__(env_name, discrete_action_space)

    def create_env(self, discrete_action_space):
        return AcrobotEnvWrapper(discrete_action_space=discrete_action_space)

    def drift(self, new_env_params):
        super(Acrobot, self).drift(new_env_params)
        self._drifted_env = copy.deepcopy(self._env).unwrapped
        self._drifted_env.LINK_LENGTH_1 = new_env_params['link_length_1']  # default is ?
        self._drifted_env.LINK_COM_POS_1 = new_env_params['link_com_pos_1']  # default is ?
        self._drifted_env.LINK_MASS_2 = new_env_params['link_mass_2']  # default is ?
        self._drifted_env.LINK_MASS_1 = new_env_params['link_mass_1']  # default is ?

    def save(self, save_dir):
        pass

    def load(self, load_dir):
        pass
