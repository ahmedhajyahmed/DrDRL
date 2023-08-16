from dr_drl.agents.ppo import PPOAgent
from dr_drl.envs.cartpole.cartpole import Cartpole
from dr_drl.runners.ppo.train_runner import TrainRunner
from dr_drl.tracker import Tracker


def main():
    # set Tracker
    tracker = Tracker(env_params_name=["masscart", "masspole", "length", "cart_friction"])

    # specify these configs along with the other agent params in yaml file
    network_config = {'units': (256, 256),
                      'lr': 0.001}

    for i in range(2):
        instance_name = 'instance_' + str(i)

        # Set up environment
        env = Cartpole('cartpole-v0', discrete_action_space=True)

        # Set up agent
        agent = PPOAgent(obs_dim=env.get_obs_dim(), act_dim=env.get_act_dim(), policy_conf=network_config,
                         discrete=True, act_high_low=env.get_act_high_low(), batch_size=128, c1=1.0, c2=0.01,
                         clip_ratio=0.2, gamma=0.98, lam=0.8, n_updates=4, layer_names=None, nb_observations=50000,
                         sparsity=0.5)
        # Set up runner
        training_runner = TrainRunner(env, agent=agent,
                                         train_episodes=600,
                                         test_episodes=100,
                                         max_steps_per_episode=200,
                                         reward_threshold=195,
                                         validation_period=10,
                                         tracker=tracker, max_reward_value=200)

        training_runner.run_training(instance_name, save_dir=instance_name)


if __name__ == '__main__':
    main()
