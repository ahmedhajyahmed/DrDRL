from dr_drl.envs.mountaincar.mountaincar import MountainCar
from dr_drl.agents.dqn import DQNAgent
from dr_drl.runners.dqn.train_runner import TrainRunner
from dr_drl.tracker import Tracker

def main():
    # set Tracker
    tracker = Tracker(env_params_name=["force", "goal_velocity", "gravity"])

    network_config = {'kernel_sizes': [256, 256],
                      'lr': 0.001}

    for i in range(3, 10):
        instance_name = 'instance_' + str(i)

        # Set up environment
        env = MountainCar('Mountaincar-v0', discrete_action_space=True)

        # Set up agent

        agent = DQNAgent(obs_dim=env.get_obs_dim(), act_dim=env.get_act_dim(), q_net_conf=network_config,
                         batch_size=128, discount_rate=0.98, min_epsilon=0.07, buffer_size=1000000,
                         target_network_update_freq=100, epsilon_decay=0.0001)

        # Set up runner
        training_runner = TrainRunner(env, agent=agent, train_episodes=2000, test_episodes=100,
                                         max_steps_per_episode=200, reward_threshold=-110, validation_period=10,
                                         tracker=tracker, max_reward_value=1)

        average_reward = training_runner.run_training(instance_name, save_dir=instance_name)
        print(average_reward)


if __name__ == '__main__':
    main()
