from dr_drl.agents.ppo import PPOAgent
from dr_drl.envs.acrobot.acrobot import Acrobot
from dr_drl.runners.ppo.train_runner import TrainingRunner
from dr_drl.tracker import Tracker


def main():
    # set Tracker
    tracker = Tracker(env_params_name=["link_com_pos_1", "link_length_1", "link_mass_1", "link_mass_2"])

    network_config = {'units': (256, 256),
                      'lr': 0.001}

    for i in range(10):
        instance_name = 'instance_' + str(i)

        # Set up environment
        env = Acrobot('acrobot-v1', discrete_action_space=True)

        # Set up agent
        agent = PPOAgent(obs_dim=env.get_obs_dim(), act_dim=env.get_act_dim(), policy_conf=network_config,
                         discrete=True, act_high_low=env.get_act_high_low(), batch_size=128, c1=1.0, c2=0.01,
                         clip_ratio=0.2, gamma=0.99, lam=0.8, n_updates=4, layer_names=None, nb_observations=50000,
                         sparsity=0.5)
        # Set up runner
        training_runner = TrainingRunner(env, agent=agent,
                                         train_episodes=2000,
                                         test_episodes=100,
                                         max_steps_per_episode=500,
                                         reward_threshold=-100,
                                         validation_period=10,
                                         tracker=tracker, max_reward_value=1)

        training_runner.run_training(instance_name, save_dir=instance_name)


if __name__ == '__main__':
    main()
