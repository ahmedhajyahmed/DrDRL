from ppo_runner.training_runner import TrainingRunner
from ppo import PPOAgent
from dr_drl.tracker import Tracker
import gym


def main():
    # set Tracker
    tracker = Tracker(env_params_name=["force", "goal_velocity", "gravity"])

    network_config = {'units': [256, 256],
                      'lr': 0.001}

    for i in range(10):
        instance_name = 'instance_' + str(i)

        # Set up environment
        # env = MountainCar('mountaincar-v0', discrete_action_space=False)
        env = gym.make('MountainCar-v0')
        obs_dim = env.observation_space.shape
        act_dim = env.action_space.n
        # Set up agent
        agent = PPOAgent(obs_dim=obs_dim, act_dim=act_dim, policy_kl_range=0.0008, policy_params=20,
                         value_clip=1.0, entropy_coef=0.05, vf_loss_coef=1.0, minibatch=2, PPO_epochs=4, gamma=0.99,
                         lam=0.95, learning_rate=2.5e-4, n_update=32, layer_names=None, nb_observations=50000,
                         sparsity=0.5, epsilon=1)
        # Set up runner
        training_runner = TrainingRunner(env, agent=agent,
                                         train_episodes=25000,
                                         test_episodes=100,
                                         max_steps_per_episode=200,
                                         reward_threshold=-110,
                                         validation_period=50,
                                         tracker=tracker, max_reward_value=1)

        training_runner.run_training(instance_name, save_dir=instance_name)


if __name__ == '__main__':
    main()
