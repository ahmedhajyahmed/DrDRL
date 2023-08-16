from dr_drl.runners.base import Runner
import numpy as np
import time


class DrDRLRunner(Runner):

    def __init__(self, env, agent, train_episodes, test_episodes, max_steps_per_episode, reward_threshold=None,
                 validation_period=10, tracker=None, seed=42, max_reward_value=None):
        super().__init__(env, agent, train_episodes, test_episodes, max_steps_per_episode, reward_threshold,
                         validation_period, tracker, seed)
        self.max_reward_value = max_reward_value

    def run_one_step(self, state, episode_metrics=None):

        action, value, log_prob = self._agent.action(state, pruning=True)  # determine action
        next_state, reward, done, _, _ = self._env.step(action)  # act on env

        return action, value, log_prob, done, next_state, reward

    def run_episode(self, episode):
        done, steps = False, 0
        state, _ = self._env.reset()
        obs, actions, log_probs, rewards, v_preds = [], [], [], [], []

        while not done and steps < self._max_steps_per_episode:
            action, value, log_prob, done, next_state, reward = self.run_one_step(state)

            rewards.append(reward)
            v_preds.append(value)
            obs.append(state)
            actions.append(action)
            log_probs.append(log_prob)

            steps += 1
            state = next_state

        return obs, actions, log_probs, rewards, v_preds, steps

    def run_n_epoch(self, episode_number, epoch):

        obs, actions, log_probs, rewards, v_preds, steps = self.run_episode(episode=episode_number)

        next_v_preds = v_preds[1:] + [0]
        gaes = self._agent.get_gaes(rewards, v_preds, next_v_preds)
        gaes = np.array(gaes).astype(dtype=np.float64)
        gaes = (gaes - gaes.mean()) / gaes.std()
        data = [obs, actions, log_probs, next_v_preds, rewards, gaes]

        for i in range(self._agent.n_updates):
            # Sample training data
            sample_indices = np.random.randint(low=0, high=len(rewards), size=self._agent.batch_size)
            sampled_data = [np.take(a=a, indices=sample_indices, axis=0) for a in data]
            # Train model
            self._agent.update(sampled_data, pruning=True)
            epoch += 1

        return epoch, rewards, steps

    def run_training(self, instance_name, save_dir=None):
        print("********************Training******************")
        start = time.time()
        self._agent.setup_pruning()
        self._agent.switch_train()
        training_metrics = {"rewards": [], "episodes": 0, "total_steps": 0}
        epoch, episodes = 0, 0
        counter = 0
        break_flag = False
        average_reward = - np.inf
        while episodes < self._train_episodes and average_reward < self._reward_threshold:
            # Run training episode
            epoch, rewards, steps = self.run_n_epoch(episodes, epoch)
            print("episode: {} / epoch: {} / score: {} /".format(episodes, epoch, np.sum(rewards)))
            episodes += 1
            counter += 1
            training_metrics["rewards"].append(np.sum(rewards))
            training_metrics["episodes"] = episodes
            training_metrics["total_steps"] += steps

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
            obs, actions, log_probs, rewards, v_preds, steps = self.run_episode(episodes)
            print("episode: {}/{}, score: {}".format(episodes, self._test_episodes, np.sum(rewards)))
            episodes += 1
            testing_metrics["rewards"].append(np.sum(rewards))

            # stopping criteria for testing
            max_expected_reward = (np.sum(np.array(testing_metrics["rewards"])) +
                                   (self.max_reward_value * (self._test_episodes - episodes))) / self._test_episodes
            if do_break and max_expected_reward < self._reward_threshold:
                return np.mean(np.array(testing_metrics["rewards"])), True

        return np.mean(np.array(testing_metrics["rewards"])), False
