from dr_drl.runners.base import Runner
import copy
import numpy as np
import time
from matplotlib import pyplot as plt
import pandas as pd


def plot_episode_stats(episode_lengths, episode_rewards, smoothing_window=5):
    # Plot the episode length over time
    fig_1 = plt.figure(figsize=(10,5))
    plt.plot(episode_lengths)
    plt.xlabel("Episode")
    plt.ylabel("Episode Length")
    plt.title("Episode Length over Time")
    # plt.show()
    fig_1.savefig('episode_length_over_time.png', dpi=fig_1.dpi)

    # Plot the episode reward over time
    fig_2 = plt.figure(figsize=(10,5))
    rewards_smoothed = pd.Series(episode_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()
    plt.plot(rewards_smoothed)
    plt.xlabel("Episode")
    plt.ylabel("Episode Reward (Smoothed)")
    plt.title("Episode Reward over Time (Smoothed over window size {})".format(smoothing_window))
    # plt.show()
    fig_2.savefig('episode_reward_over_time.png', dpi=fig_2.dpi)

    # Plot time steps and episode number
    fig_3 = plt.figure(figsize=(10,5))
    plt.plot(np.cumsum(episode_lengths), np.arange(len(episode_lengths)))
    plt.xlabel("Time Steps")
    plt.ylabel("Episode")
    plt.title("Episode per time step")
    # plt.show()
    fig_3.savefig('episode_per_time_step.png', dpi=fig_3.dpi)


def plot_reward(episode_rewards):
    fig_4 = plt.figure(figsize=(10, 5))
    plt.plot(range(len(episode_rewards)), episode_rewards)
    plt.xlabel("Episode")
    plt.ylabel("Episode Reward")
    plt.title("Episode Reward over Time")
    # plt.show()
    fig_4.savefig('episode_reward_over_time.png', dpi=fig_4.dpi)


class TrainRunner(Runner):

    def __init__(self, env, agent, train_episodes, test_episodes, max_steps_per_episode,
                 reward_threshold=None, validation_period=10, tracker=None,
                 seed=42, max_reward_value=None):
        super().__init__(env, agent, train_episodes, test_episodes, max_steps_per_episode, reward_threshold,
                         validation_period, tracker, seed)
        self.max_reward_value = max_reward_value
        self.check = 1
        # Parameters for the consecutive actions technique
        self.cons_acts = 4
        self.prob_act = 0.5
        self.episode_lengths = [0] * self._max_steps_per_episode
        self.episode_rewards = [0] * self._max_steps_per_episode

    def run_one_step(self, state, episode_metrics):

        action = self._agent.action(state)

        # Techniques to force exploration, useful in sparse rewards environments ####

        # Using the consecutive steps technique
        if self.check == 1 and np.random.uniform() < self.prob_act:
            # print(self.replay_buffer.n_entries)
            for i in range(self.cons_acts):
                self._env.step(action)

        '''
        # Using OUNoise technique + epsilon-greedy
        if np.random.uniform() < epsilon:
            action = noise.get_action(action, k)
        if check==0 and epsilon > epsilon_min:
            epsilon = epsilon * epsilon_dk
        '''
        ################################################################################

        new_state, reward, done, turn, info = self._env.step(action)
        # new_state, reward, done, _ = self._env.step(action)
        new_state = new_state.astype(np.float32)

        if self._agent._training:
            self._agent.replay_buffer.add(state, action, reward, new_state, done)

            self.check = self._agent.update(check=self.check)  # train models through memory replay

        episode_metrics["reward"] += reward
        episode_metrics["steps"] += 1

        return done, new_state, episode_metrics

    def run_episode(self, episodes, total_steps=0, train=True):
        episode_metrics = {"reward": 0, "steps": 0, "epsilon": 0}
        done = False
        state, _ = self._env.reset()
        # state = self._env.reset().astype(np.float32)
        steps = 0
        use_random = False
        # Run the loop until the episode ends or reach a specific condition
        while not done and steps < self._max_steps_per_episode:

            done, state, episode_metrics = self.run_one_step(state, episode_metrics)
            steps += 1

        if train:
            self.episode_lengths[episodes] = steps
            self.episode_rewards[episodes] = episode_metrics["reward"]

        total_episodes = self._train_episodes if train else self._test_episodes
        print("episode: {}/{}, reward: {}"
              .format(episodes, total_episodes, episode_metrics["reward"]))

        return episode_metrics, total_steps + steps

    def run_training(self, instance_name, save_dir=None):
        print("********************Training******************")
        start = time.time()
        self._agent.switch_train()

        training_metrics = {"rewards": [], "episodes": 0, "total_steps": 0}
        total_steps = 0
        episodes = 0
        counter = 0
        break_flag = False
        average_reward = - np.inf
        while episodes <= self._train_episodes and average_reward < self._reward_threshold:
            # Run training episode
            episode_metrics, total_steps = self.run_episode(episodes, total_steps=total_steps)
            episodes += 1
            counter += 1
            training_metrics["rewards"].append(episode_metrics["reward"])
            training_metrics["episodes"] = episodes
            training_metrics["total_steps"] += episode_metrics["steps"]

            # stopping criteria
            if self._run_testing and counter > self._validation_period and \
                    np.average(training_metrics["rewards"][-self._validation_period:]) >= self._reward_threshold:
                # perform test
                average_reward, break_flag = self.run_testing(do_break=True)
                self._agent.switch_train()
                counter = 0

        if average_reward == - np.inf or break_flag:
            # perform test
            average_reward, _ = self.run_testing()

        if save_dir:
            self._agent.save(save_dir)

        if not self._env._is_drifted:
            self._tracker.add_training_row([instance_name, time.time() - start, episodes, average_reward])
            self._tracker.save_training()
        else:
            self._tracker.add_continual_learning_data([time.time() - start, episodes, training_metrics["total_steps"],
                                                       average_reward])

        # self._tracker.add_training_row([instance_name, time.time() - start, episodes, average_reward])
        # self._tracker.save_training()

        # plot_episode_stats(self.episode_lengths, self.episode_rewards)
        # plot_reward(self.episode_rewards)

        return average_reward

    def run_testing(self, do_break=False):
        print("********************Testing******************")
        self._agent.switch_test()
        testing_metrics = {"rewards": []}
        episodes = 0
        while episodes <= self._test_episodes:
            # Run test episodes
            episode_metrics, _ = self.run_episode(episodes, train=False)
            episodes += 1
            testing_metrics["rewards"].append(episode_metrics["reward"])

            # stopping criteria for testing
            max_expected_reward = (np.sum(np.array(testing_metrics["rewards"])) +
                                   (self.max_reward_value * (self._test_episodes - episodes))) / self._test_episodes
            if do_break and max_expected_reward < self._reward_threshold:
                return np.mean(np.array(testing_metrics["rewards"])), True

        return np.mean(np.array(testing_metrics["rewards"])), False
