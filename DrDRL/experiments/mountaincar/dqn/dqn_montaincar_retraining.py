import copy
import numpy as np
from dr_drl.envs.mountaincar.mountaincar import MountainCar
from dr_drl.agents.dqn import DQNAgent
from dr_drl.envs.mountaincar.drift_params.dqn import adaptable_env, non_adaptable_env
from dr_drl.runners.dqn.train_runner import TrainRunner
from dr_drl.runners.dqn.dr_drl_runner import DrDRLRunner
from dr_drl.tracker import Tracker


def perform_environment_drift(runner, drift_type="adaptable"):
    failed_to_solve_env = False

    while not failed_to_solve_env:
        print("------------------drifting the environment---------------------")
        if drift_type == "soft":
            new_env_params = copy.deepcopy(adaptable_env[np.random.randint(len(adaptable_env))])
        else:
            new_env_params = copy.deepcopy(non_adaptable_env[np.random.randint(len(non_adaptable_env))])
            multiplier = np.random.uniform(low=1, high=2)
            for el in new_env_params:
                new_env_params[el] = new_env_params[el] * multiplier

        runner._env.drift(new_env_params)

        # perform test on the new environment
        average_reward, _ = runner.run_testing(do_break=True)
        print("average reward: ", average_reward)
        failed_to_solve_env = runner._reward_threshold > average_reward


def main():
    reward_tolerance_rate = 0.1
    reward_threshold = -110
    tolerated_reward = reward_threshold - int(abs(reward_threshold) * reward_tolerance_rate)
    # set Tracker
    tracker = Tracker(env_params_name=["force", "goal_velocity", "gravity"], to_train=False)

    # specify these configs along with the other agent params in yaml file
    network_config = {'kernel_sizes': [256, 256],
                      'lr': 0.001}

    for i in range(10):
        instance_name = 'instance_' + str(i)

        # Set up environment
        for j in range(6):
            iteration_seed = np.random.randint(1000000)
            full_index = str(i) + "_" + str(j)

            env = MountainCar('mountaincar-v0', discrete_action_space=True)
            agent_cl = DQNAgent(obs_dim=env.get_obs_dim(), act_dim=env.get_act_dim(), q_net_conf=network_config,
                                batch_size=128, discount_rate=0.98, min_epsilon=0.07, buffer_size=1000000,
                                target_network_update_freq=100, epsilon_decay=0.001, load_buffer=True)
            agent_cl.load(instance_name + "/", load_observation=False)

            agent_pr = DQNAgent(obs_dim=env.get_obs_dim(), act_dim=env.get_act_dim(), q_net_conf=network_config,
                                batch_size=128, discount_rate=0.98, min_epsilon=0.07, buffer_size=1000000,
                                target_network_update_freq=100, epsilon_decay=0.001, sparsity=0.9, f_epsilon=0.0001,
                                load_buffer=True)
            agent_pr.load(instance_name + "/")

            # Set up runner
            continual_learning_runner = TrainRunner(env, agent=agent_cl, train_episodes=300, test_episodes=100,
                                                       max_steps_per_episode=200, reward_threshold=tolerated_reward,
                                                       validation_period=5, tracker=tracker, seed=iteration_seed,
                                                       max_reward_value=1)

            if j % 2 == 0:
                drift_type = "soft"
                perform_environment_drift(continual_learning_runner, drift_type="adaptable")
            else:
                drift_type = "aggressive"
                perform_environment_drift(continual_learning_runner, drift_type="non-adaptable")
            # Set up agent

            print("------------------continual learning" + full_index + "---------------------")

            continual_learning_runner.run_training(instance_name,
                                                   save_dir=instance_name + "/" + str(j) + "/" + "cl")
            print("------------------forgetting mechanism" + full_index + "---------------------")
            # Set up runner
            pruning_retraining_runner = DrDRLRunner(continual_learning_runner._env, agent=agent_pr,
                                                                train_episodes=300, test_episodes=100,
                                                                max_steps_per_episode=200,
                                                                reward_threshold=tolerated_reward, validation_period=5,
                                                                tracker=tracker, seed=iteration_seed,
                                                                max_reward_value=1)
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
