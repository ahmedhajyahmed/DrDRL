from dr_drl.agents.sac import SACAgent
from dr_drl.envs.mountaincarcontinuous.mountaincarcontinuous import MountainCarContinuous
from dr_drl.runners.sac.train_runner import TrainRunner
from dr_drl.tracker import Tracker
import numpy as np
import gym


def main():
    # set Tracker
    tracker = Tracker(env_params_name=["power", "goal_velocity"])

    for i in range(10):
        instance_name = 'instance_' + str(i)

        env = MountainCarContinuous('mountaincarcontinuous-v0', discrete_action_space=False)

        obs_dim = env._env.observation_space.shape[0]
        n_actions = env._env.action_space.shape[0]
        act_lim = env._env.action_space.high

        # Set up agent
        agent = SACAgent(obs_dim=obs_dim, act_dim=n_actions, act_lim=act_lim, discount=0.99, temperature=0.3,
                         polyak_coef=0.01, lr=1e-3, hidden_layers=2, n_hidden_units=60, batch_size=128,
                         replay_start_size=10000, buffer_size=int(1e6), layer_names=None, nb_observations=50000,
                         sparsity=0.5, epsilon=1, load_buffer=True)
        # Set up runner
        training_runner = TrainRunner(env, agent=agent,
                                         train_episodes=100,
                                         test_episodes=100,
                                         max_steps_per_episode=1000,
                                         reward_threshold=90,
                                         validation_period=10,
                                         tracker=tracker, max_reward_value=1)

        training_runner.run_training(instance_name, save_dir=instance_name)


if __name__ == '__main__':
    main()
