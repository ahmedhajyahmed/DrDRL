from dr_drl.runners.base import Runner
import copy
import numpy as np
import time


class DrDRLRunner(Runner):

    def __init__(self, env, agent, train_episodes, test_episodes, max_steps_per_episode, reward_threshold=None,
                 validation_period=10, tracker=None, seed=42, max_reward_value=None):
        super().__init__(env, agent, train_episodes, test_episodes, max_steps_per_episode, reward_threshold,
                         validation_period, tracker, seed)
        self.max_reward_value = max_reward_value

    def run_one_step(self, state, episode_metrics, pruning=True):

        action = self._agent.action(state, pruning=pruning)
        next_state, reward, done, turn, info = self._env.step(action)

        data = state, action, reward, next_state, done
        epsilon = self._agent.update(copy.deepcopy(data), pruning=pruning)

        episode_metrics["reward"] += reward
        episode_metrics["steps"] += 1
        episode_metrics["epsilon"] = epsilon

        return done, next_state, episode_metrics

    def run_episode(self, episodes, train=True, pruning=True):
        episode_metrics = {"reward": 0, "steps": 0, "epsilon": 0}
        done = False
        state, _ = self._env.reset()
        steps = 0
        # Run the loop until the episode ends or reach a specific condition
        while not done and steps < self._max_steps_per_episode:
            done, state, episode_metrics = self.run_one_step(state, episode_metrics, pruning=pruning)
            steps += 1

        total_episodes = self._train_episodes if train else self._test_episodes
        print("episode: {}/{}, score: {}, e: {}"
              .format(episodes, total_episodes, episode_metrics["reward"], episode_metrics["epsilon"]))

        return episode_metrics

    def run_training(self, instance_name, save_dir=None, pruning=True):
        print("********************Training******************")
        start = time.time()
        self._agent.setup_pruning()
        self._agent.switch_train()
        training_metrics = {"rewards": [], "episodes": 0, "steps": [], "epsilons": []}
        episodes = 0
        counter = 0
        break_flag = False
        average_reward = - np.inf
        while episodes <= self._train_episodes and average_reward < self._reward_threshold:
            # Run training episode
            episode_metrics = self.run_episode(episodes, pruning=pruning)
            episodes += 1
            counter += 1
            training_metrics["rewards"].append(episode_metrics["reward"])
            training_metrics["episodes"] = episodes
            training_metrics["steps"].append(episode_metrics["steps"])
            training_metrics["epsilons"].append(episode_metrics["epsilon"])

            # stopping criteria
            # if self._run_testing and len(training_metrics["rewards"]) > self._validation_period and \
            #         np.average(training_metrics["rewards"][-self._validation_period:]) > self._reward_threshold:
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

        self._tracker.add_pruning_retraining_data([time.time() - start, episodes,
                                                   average_reward])
        return average_reward, training_metrics

    def run_testing(self, do_break=False, pruning=True):
        print("********************Testing******************")
        self._agent.switch_test()
        testing_metrics = {"rewards": []}
        episodes = 0
        while episodes <= self._test_episodes:
            # Run test episodes
            episode_metrics = self.run_episode(episodes, train=False, pruning=pruning)
            episodes += 1
            testing_metrics["rewards"].append(episode_metrics["reward"])

            # stopping criteria for testing
            max_expected_reward = (np.sum(np.array(testing_metrics["rewards"])) +
                                   (self.max_reward_value * (self._test_episodes - episodes))) / self._test_episodes
            if do_break and max_expected_reward < self._reward_threshold:
                return np.mean(np.array(testing_metrics["rewards"])), True

        return np.mean(np.array(testing_metrics["rewards"])), False

    def collect_drifted_observations(self):
        print("********************collecting observations******************")
        self._agent.switch_test()
        episodes = 0
        while episodes <= self._test_episodes:
            episode_reward = 0
            done = False
            state, _ = self._env.reset()
            steps = 0
            # Run the loop until the episode ends or reach a specific condition
            while not done and steps < self._max_steps_per_episode:
                self._agent._region_locator.add_observations(state)
                action = self._agent.action(state, pruning=False)
                next_state, reward, done, turn, info = self._env.step(action)
                episode_reward += reward
                steps += 1
                state = next_state

            total_episodes = self._test_episodes
            print("episode: {}/{}, score: {}".format(episodes, total_episodes, episode_reward))
            episodes += 1
        self._agent._region_locator.observations = np.vstack(self._agent._region_locator.observations)
        self._agent._region_locator.set_model(self._agent._q_net)
        self._agent._region_locator.set_layer_names([self._agent._q_net.layers[0].name,
                                                     self._agent._q_net.layers[1].name])
        print()
