import copy
from dr_drl.envs.cartpole.cartpole_wrapper import CartPoleEnvWrapper
from dr_drl.envs.base_env import BaseEnv


class Cartpole(BaseEnv):

    def __init__(self, env_name, discrete_action_space=True):
        super().__init__(env_name, discrete_action_space)

    def create_env(self, discrete_action_space):
        return CartPoleEnvWrapper(discrete_action_space=discrete_action_space)

    def drift(self, new_env_params):
        super(Cartpole, self).drift(new_env_params)
        self._drifted_env = copy.deepcopy(self._env).unwrapped
        self._drifted_env.masscart = new_env_params['masscart']  # default is 1
        self._drifted_env.masspole = new_env_params['masspole']  # default is 0.1
        self._drifted_env.length = new_env_params['length']  # default is 0.5
        self._drifted_env.cart_friction = new_env_params['cart_friction']  # default is 0.0

    def save(self, save_dir):
        pass

    def load(self, load_dir):
        pass



