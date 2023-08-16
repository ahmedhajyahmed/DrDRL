from experiments.mountaincar.ppo.ppo import PPOAgent
from ppo_runner.pruning_retraining_runner import DrDRLRunner
from dr_drl.tracker import Tracker
import copy
import gym
import numpy as np
from dr_drl.envs.mountaincar.drift_params.ppo import adaptable_env, non_adaptable_env
from ppo_runner.training_runner import TrainingRunner


def drift_env(env, new_env_params):
    drifted_env = copy.deepcopy(env).unwrapped
    drifted_env.force = new_env_params['force']  # default is ?
    drifted_env.gravity = new_env_params['gravity']  # default is ?
    drifted_env.goal_velocity = new_env_params['goal_velocity']
    return drifted_env

def perform_environment_drift(runner, drift_type="soft"):
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

        # runner._env.drift(new_env_params)
        drifted_env = drift_env(runner._env, new_env_params)
        runner.set_env(drifted_env)

        # perform test on the new environment
        average_reward, _ = runner.run_testing(do_break=True)
        print("average reward: ", average_reward)
        # tolerated_reward = runner._reward_threshold - int(
        #     abs(runner._reward_threshold) * reward_tolerance_rate)
        failed_to_solve_env = runner._reward_threshold > average_reward > - runner._max_steps_per_episode
        print()

    return drifted_env, new_env_params


def main():
    reward_tolerance_rate = 0.1
    reward_threshold = -110
    tolerated_reward = reward_threshold - int(abs(reward_threshold) * reward_tolerance_rate)
    # set Tracker
    tracker = Tracker(env_params_name=["force", "goal_velocity", "gravity"], to_train=False)

    # specify these configs along with the other agent params in yaml file
    network_config = {'kernel_sizes': [256, 256],
                      'lr': 0.001}

    for i in range(8, 10):
        instance_name = 'instance_' + str(i)

        # Set up environment
        for j in range(6):
            iteration_seed = np.random.randint(1000000)
            full_index = str(i) + "_" + str(j)

            # Set up environment
            # env = MountainCar('mountaincar-v0', discrete_action_space=True)
            env = gym.make('MountainCar-v0')
            obs_dim = env.observation_space.shape
            act_dim = env.action_space.n
            # Set up agent
            agent_cl = PPOAgent(obs_dim=obs_dim, act_dim=act_dim, policy_kl_range=0.0008, policy_params=20,
                                value_clip=1.0, entropy_coef=0.05, vf_loss_coef=1.0, minibatch=2, PPO_epochs=4,
                                gamma=0.99, lam=0.95, learning_rate=2.5e-4, n_update=32, layer_names=None,
                                nb_observations=50000, sparsity=0.5)
            agent_cl.load(instance_name + "/", load_observation=False)
            # Set up agent
            agent_pr = PPOAgent(obs_dim=obs_dim, act_dim=act_dim, policy_kl_range=0.0008, policy_params=20,
                                value_clip=1.0, entropy_coef=0.05, vf_loss_coef=1.0, minibatch=2, PPO_epochs=4,
                                gamma=0.99, lam=0.95, learning_rate=2.5e-4, n_update=32, layer_names=None,
                                nb_observations=50000, sparsity=0.9, epsilon=0.0001)
            agent_pr.load(instance_name + "/", load_pruned=True)

            # Set up runner
            continual_learning_runner = TrainingRunner(env, agent=agent_cl, train_episodes=9000, test_episodes=100,
                                                       max_steps_per_episode=200, reward_threshold=tolerated_reward,
                                                       validation_period=10, tracker=tracker, seed=iteration_seed,
                                                       max_reward_value=1)

            if j % 2 == 0:
                drift_type = "soft"
                drifted_env, params = perform_environment_drift(continual_learning_runner, drift_type= drift_type)
            else:
                drift_type = "aggressive"
                drifted_env, params = perform_environment_drift(continual_learning_runner, drift_type= drift_type)
            # Set up agent

            print("------------------continual learning" + full_index + "---------------------")

            continual_learning_runner.set_env(drifted_env)
            continual_learning_runner.run_training(instance_name,
                                                   save_dir=instance_name + "/" + str(j) + "/" + "cl")
            print("------------------forgetting mechanism" + full_index + "---------------------")
            # Set up runner
            pruning_retraining_runner = DrDRLRunner(drifted_env, agent=agent_pr,
                                                                train_episodes=9000, test_episodes=100,
                                                                max_steps_per_episode=200,
                                                                reward_threshold=tolerated_reward, validation_period=10,
                                                                tracker=tracker, seed=iteration_seed,
                                                                max_reward_value=1)
            pruning_retraining_runner.run_training(instance_name,
                                                   save_dir=instance_name + "/" + str(j) + "/" + "pr")

            row = [full_index] + list(params.values())
            for index in range(len(tracker.continual_learning_data)):
                row += [tracker.continual_learning_data[index], tracker.pruning_retraining_data[index]]
            row.append(drift_type)
            row.append(str(iteration_seed))

            tracker.add_repair_row(row=row)
            tracker.save_repair()

        tracker.save_repair()


if __name__ == '__main__':
    main()
