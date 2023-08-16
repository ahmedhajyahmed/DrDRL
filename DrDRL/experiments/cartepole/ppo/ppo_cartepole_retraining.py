from dr_drl.agents.ppo import PPOAgent
from dr_drl.envs.cartpole.cartpole import Cartpole
from dr_drl.runners.ppo.train_runner import TrainRunner
from dr_drl.tracker import Tracker
import copy
import numpy as np
from dr_drl.envs.cartpole.drift_params.ppo import adaptable_env, non_adaptable_env
from dr_drl.runners.ppo.dr_drl_runner import DrDRLRunner


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
    reward_tolerance_rate = 0
    reward_threshold = 195
    tolerated_reward = reward_threshold - int(reward_threshold * reward_tolerance_rate)
    # set Tracker
    tracker = Tracker(env_params_name=["masscart", "masspole", "length", "cart_friction"], to_train=False)

    # specify these configs along with the other agent params in yaml file
    network_config = {'units': (256, 256),
                      'lr': 0.001}

    for i in range(10):
        instance_name = 'instance_' + str(i)

        # Set up environment
        for j in range(6):
            iteration_seed = np.random.randint(1000000)
            full_index = str(i) + "_" + str(j)

            # Set up environment
            env = Cartpole('cartpole-v0', discrete_action_space=True)
            # Set up agent
            agent_cl = PPOAgent(obs_dim=env.get_obs_dim(), act_dim=env.get_act_dim(), policy_conf=network_config,
                                discrete=True, act_high_low=env.get_act_high_low(), batch_size=128, c1=1.0, c2=0.01,
                                clip_ratio=0.2, gamma=0.98, lam=0.8, n_updates=4, layer_names=None,
                                nb_observations=50000, sparsity=0.5)
            agent_cl.load(instance_name + "/", load_observation=False)
            # Set up agent
            agent_pr = PPOAgent(obs_dim=env.get_obs_dim(), act_dim=env.get_act_dim(), policy_conf=network_config,
                                discrete=True, act_high_low=env.get_act_high_low(), batch_size=128, c1=1.0, c2=0.01,
                                clip_ratio=0.2, gamma=0.98, lam=0.8, n_updates=4, layer_names=None,
                                nb_observations=50000, sparsity=0.5, epsilon=0.001)
            agent_pr.load(instance_name + "/")

            # Set up runner
            continual_learning_runner = TrainRunner(env, agent=agent_cl, train_episodes=200, test_episodes=100,
                                                       max_steps_per_episode=200, reward_threshold=tolerated_reward,
                                                       validation_period=10, tracker=tracker, seed=iteration_seed,
                                                       max_reward_value=200)

            if j % 2 == 0:
                drift_type = "soft"
                perform_environment_drift(continual_learning_runner, drift_type=drift_type)
            else:
                drift_type = "aggressive"
                perform_environment_drift(continual_learning_runner, drift_type=drift_type)
            # Set up agent

            print("------------------continual learning" + full_index + "---------------------")

            continual_learning_runner.run_training(instance_name,
                                                   save_dir=instance_name + "/" + str(j) + "/" + "cl")
            print("------------------forgetting mechanism" + full_index + "---------------------")
            # Set up runner
            dr_drl_runner = DrDRLRunner(continual_learning_runner._env, agent=agent_pr,
                                                                train_episodes=200, test_episodes=100,
                                                                max_steps_per_episode=200,
                                                                reward_threshold=tolerated_reward, validation_period=10,
                                                                tracker=tracker, seed=iteration_seed,
                                                                max_reward_value=200)
            dr_drl_runner.run_training(instance_name,
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
