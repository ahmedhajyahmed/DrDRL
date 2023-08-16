import copy
from dr_drl.agents.sac import SACAgent
from dr_drl.envs.mountaincarcontinuous.mountaincarcontinuous import MountainCarContinuous
from dr_drl.envs.mountaincarcontinuous.drift_params.sac import adaptable_env, non_adaptable_env
from dr_drl.runners.sac.dr_drl_runner import DrDRLRunner
from dr_drl.runners.sac.train_runner import TrainRunner
from dr_drl.tracker import Tracker
import numpy as np


def perform_environment_drift(runner, drift_type="soft"):
    failed_to_solve_env = False

    while not failed_to_solve_env:
        print("------------------drifting the environment---------------------")
        if drift_type == "soft":
            new_env_params = copy.deepcopy(adaptable_env[np.random.randint(len(adaptable_env))])
        else:
            new_env_params = copy.deepcopy(non_adaptable_env[np.random.randint(len(non_adaptable_env))])
            multiplier = np.random.uniform(low=1, high=5)
            for el in new_env_params:
                new_env_params[el] = new_env_params[el] * multiplier

        runner._env.drift(new_env_params)

        # perform test on the new environment
        average_reward, _ = runner.run_testing()
        print("average reward: ", average_reward)
        failed_to_solve_env = runner._reward_threshold > average_reward


def main():
    reward_tolerance_rate = 0.2
    reward_threshold = 90
    tolerated_reward = reward_threshold - int(abs(reward_threshold) * reward_tolerance_rate)

    # set Tracker
    tracker = Tracker(env_params_name=["power", "goal_velocity", "gravity"], to_train=False)

    for i in range(10):
        instance_name = 'instance_' + str(i)

        for j in range(6):
            iteration_seed = np.random.randint(1000000)
            full_index = str(i) + "_" + str(j)

            env = MountainCarContinuous('mountaincarcontinuous-v0', discrete_action_space=False)

            obs_dim = env._env.observation_space.shape[0]
            n_actions = env._env.action_space.shape[0]
            act_lim = env._env.action_space.high

            agent_cl = SACAgent(obs_dim=obs_dim, act_dim=n_actions, act_lim=act_lim, discount=0.99, temperature=0.3,
                                polyak_coef=0.01, lr=1e-3, hidden_layers=2, n_hidden_units=60, batch_size=128,
                                replay_start_size=10000, buffer_size=int(1e6), layer_names=None, nb_observations=50000,
                                sparsity=0.5, epsilon=1, load_buffer=True)
            agent_cl.load(instance_name + "/", load_observation=False)

            agent_pr = SACAgent(obs_dim=obs_dim, act_dim=n_actions, act_lim=act_lim, discount=0.99, temperature=0.3,
                                polyak_coef=0.01, lr=1e-3, hidden_layers=2, n_hidden_units=60, batch_size=128,
                                replay_start_size=10000, buffer_size=int(1e6), layer_names=None, nb_observations=50000,
                                sparsity=0.9, epsilon=0.0001, load_buffer=False)
            agent_pr.load(instance_name + "/", load_pruned=True)

            # Set up runner
            continual_learning_runner = TrainRunner(env, agent=agent_cl,
                                                       train_episodes=100,
                                                       test_episodes=100,
                                                       max_steps_per_episode=1000,
                                                       reward_threshold=tolerated_reward,
                                                       validation_period=5,
                                                       seed=iteration_seed,
                                                       tracker=tracker, max_reward_value=90)

            if j % 2 == 0:
                drift_type = "soft"
                perform_environment_drift(continual_learning_runner, drift_type=drift_type)
            else:
                drift_type = "aggressive"
                perform_environment_drift(continual_learning_runner, drift_type= drift_type)

            print("------------------continual learning" + full_index + "---------------------")
            continual_learning_runner.run_training(instance_name,
                                                   save_dir=instance_name + "/" + str(j) + "/" + "cl")
            print("------------------forgetting mechanism" + full_index + "---------------------")
            # Set up runner
            pruning_retraining_runner = PruningRetrainingRunner(continual_learning_runner._env, agent=agent_pr,
                                                                train_episodes=100,
                                                                test_episodes=100,
                                                                max_steps_per_episode=1000,
                                                                reward_threshold=tolerated_reward,
                                                                validation_period=5,
                                                                seed=iteration_seed,
                                                                tracker=tracker, max_reward_value=90)
            pruning_retraining_runner.run_training(instance_name,
                                                   save_dir=instance_name + "/" + str(j) + "/" + "pr")

            row = [full_index] + list(continual_learning_runner._env._drifted_env_params.values())
            for index in range(len(tracker.continual_learning_data)):
                row += [tracker.continual_learning_data[index], tracker.pruning_retraining_data[index]]
            row.append(drift_type)
            row.append(str(iteration_seed))

            tracker.add_repair_row(row=row)
            tracker.save_repair()

        tracker.save_repair()


if __name__ == '__main__':
    main()
