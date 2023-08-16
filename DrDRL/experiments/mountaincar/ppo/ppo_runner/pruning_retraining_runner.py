from dr_drl.runners.base import Runner
import numpy as np
import time


class DrDRLRunner(Runner):

    def __init__(self, env, agent, train_episodes, test_episodes, max_steps_per_episode, reward_threshold=None,
                 validation_period=10, tracker=None, seed=42, max_reward_value=None):
        super().__init__(env, agent, train_episodes, test_episodes, max_steps_per_episode, reward_threshold,
                         validation_period, tracker, seed)
        self.max_reward_value = max_reward_value

    def run_one_step(self, state, episode_metrics):

        action = int(self._agent.action(state, pruning=True))  # determine action
        next_state, reward, done, _ = self._env.step(action)  # act on env

        episode_metrics["t_updates"] += 1

        if self._agent._training:
            self._agent.save_eps(state.tolist(), action, reward, float(done), next_state.tolist())
            if episode_metrics["t_updates"] % self._agent.n_update == 0:
                batch_size = int(len(self._agent.memory) / self._agent.minibatch)
                # Optimize policy for K epochs:
                for _ in range(self._agent.PPO_epochs):
                    for states, actions, rewards, dones, next_states in self._agent.memory.get_all_items().batch(batch_size):
                        data = states, actions, rewards, dones, next_states
                        self._agent.update(data, pruning=True)

                # Clear the memory
                self._agent.memory.clear_memory()

                # Copy new weights into old policy:
                self._agent.actor_old.set_weights(self._agent.actor.get_weights())
                self._agent.critic_old.set_weights(self._agent.critic.get_weights())

                episode_metrics["t_updates"] = 0

        episode_metrics["reward"] += reward
        episode_metrics["steps"] += 1

        return done, next_state, episode_metrics

    def run_episode(self, episode, t_updates=0, train=True):
        episode_metrics = {"reward": 0, "steps": 0, "t_updates": t_updates}
        done = False
        state = self._env.reset()
        steps = 0
        # Run the loop until the episode ends or reach a specific condition
        while not done and steps < self._max_steps_per_episode:
            done, state, episode_metrics = self.run_one_step(state, episode_metrics)
            steps += 1

        return episode_metrics, episode_metrics["t_updates"]

    def run_training(self, instance_name, save_dir=None):
        print("********************Training******************")
        start = time.time()
        self._agent.setup_pruning()
        self._agent.switch_train()
        training_metrics = {"rewards": [], "episodes": 0, "total_steps": 0}
        episodes = 0
        t_updates = 0
        counter = 0
        break_flag = False
        average_reward = - np.inf
        while episodes < self._train_episodes and average_reward < self._reward_threshold:
            # Run training episode
            episode_metrics, t_updates = self.run_episode(episodes, t_updates=t_updates)
            print('Episode {} \t t_reward: {} '.format(episodes, episode_metrics["reward"]))
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
            print('Episode {} \t t_reward: {} '.format(episodes, episode_metrics["reward"]))
            episodes += 1
            testing_metrics["rewards"].append(episode_metrics["reward"])

            # stopping criteria for testing
            max_expected_reward = (np.sum(np.array(testing_metrics["rewards"])) +
                                   (self.max_reward_value * (self._test_episodes - episodes))) / self._test_episodes
            if do_break and max_expected_reward < self._reward_threshold:
                return np.mean(np.array(testing_metrics["rewards"])), True

        return np.mean(np.array(testing_metrics["rewards"])), False

    def set_env(self, env):
        self._env = env
