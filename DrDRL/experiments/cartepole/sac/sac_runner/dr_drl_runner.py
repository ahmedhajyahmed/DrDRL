from dr_drl.runners.base import Runner
import numpy as np
import time


class DrDRLRunner(Runner):

    def __init__(self, env, agent, train_episodes, test_episodes, max_steps_per_episode, n_updates,
                 random_steps=1000, reward_threshold=None, validation_period=10, tracker=None,
                 seed=42, max_reward_value=None):
        super().__init__(env, agent, train_episodes, test_episodes, max_steps_per_episode, reward_threshold,
                         validation_period, tracker, seed)
        self.n_updates = n_updates
        self.random_steps = random_steps
        self.max_reward_value = max_reward_value

    def run_one_step(self, state, episode_metrics, use_random=False):

        action = self._agent.action(state, use_random=use_random, pruning=True)
        next_state, reward, done, turn, info = self._env.step(action[0])

        self._agent.remember(state, action, reward, next_state, done)  # add to memory

        self._agent.update(pruning=True)  # train models through memory replay
        self._agent.update_target_weights(self._agent.critic_1, self._agent.critic_target_1,
                                          tau=self._agent.tau)  # iterates target model
        self._agent.update_target_weights(self._agent.critic_2, self._agent.critic_target_2, tau=self._agent.tau)

        episode_metrics["reward"] += reward
        episode_metrics["steps"] += 1

        return done, next_state, episode_metrics

    def run_episode(self, episodes, total_steps=0, train=True):
        episode_metrics = {"reward": 0, "steps": 0, "epsilon": 0}
        done = False
        state, _ = self._env.reset()
        steps = 0
        use_random = False
        # Run the loop until the episode ends or reach a specific condition
        while not done and steps < self._max_steps_per_episode:

            if train and total_steps + steps <= self.random_steps or len(self._agent.memory) <= self._agent.batch_size:
                use_random = True

            done, state, episode_metrics = self.run_one_step(state, episode_metrics, use_random=use_random)
            steps += 1

        total_episodes = self._train_episodes if train else self._test_episodes
        print("episode: {}/{}, score: {}, alpha: {}"
              .format(episodes, total_episodes, episode_metrics["reward"], self._agent.alpha.numpy()))

        return episode_metrics, total_steps + steps

    def run_training(self, instance_name, save_dir=None):
        print("********************Training******************")
        start = time.time()
        self._agent.setup_pruning()
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

        self._tracker.add_pruning_retraining_data([time.time() - start, episodes, training_metrics["total_steps"],
                                                   average_reward])
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
