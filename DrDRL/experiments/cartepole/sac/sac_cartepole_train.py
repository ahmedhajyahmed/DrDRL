from sac import SACAgent
from dr_drl.envs.cartpole.cartpole import Cartpole
from experiments.cartepole.sac.sac_runner.training_runner import TrainingRunner
from dr_drl.tracker import Tracker


def main():
    # set Tracker
    tracker = Tracker(env_params_name=["masscart", "masspole", "length", "cart_friction"])

    for i in range(2):
        instance_name = 'instance_' + str(i)

        # Set up environment
        env = Cartpole('cartpole-v0', discrete_action_space=False)

        # Set up agent
        agent = SACAgent(obs_dim=env.get_obs_dim(), act_dim=env.get_act_dim(), act_shape=env.get_act_shape(),
                         act_high_low=env.get_act_high_low(), lr_actor=3e-5, lr_critic=3e-4, actor_units=(256, 256),
                         critic_units=(256, 256), auto_alpha=True, alpha=0.2, tau=0.005, gamma=0.99, batch_size=128,
                         memory_cap=100000, layer_names=None, nb_observations=50000, sparsity=0.5)
        # Set up runner
        training_runner = TrainingRunner(env, agent=agent,
                                         train_episodes=1000,
                                         test_episodes=100,
                                         random_steps=1000,
                                         max_steps_per_episode=200,
                                         n_updates=4,
                                         reward_threshold=195,
                                         validation_period=10,
                                         tracker=tracker, max_reward_value=200)

        training_runner.run_training(instance_name, save_dir=instance_name)


if __name__ == '__main__':
    main()
