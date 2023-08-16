"""
This code was udapted from
https://github.com/chandar-lab/RLHive
"""
import os
from dr_drl.agents.agent import Agent
from dr_drl.agents.utils.replay_buffer import ReplayBuffer
from dr_drl.dnn_regions_locator import DNNRegionsLocator
import tensorflow as tf
import numpy as np
import gym


class DQNAgent(Agent):
    def __init__(self, obs_dim, act_dim, q_net_conf, batch_size=64, discount_rate=0.99, epsilon=1.0, min_epsilon=0.01,
                 epsilon_decay=0.001, buffer_size=100000, test_epsilon=0, min_replay_history=0,
                 target_network_update_freq=100, train_freq=4, layer_names=None, nb_observations=50000, sparsity=0.5,
                 f_epsilon=1, load_buffer=True):
        super().__init__(obs_dim, act_dim)
        # DNN related params
        self._q_net_conf = q_net_conf
        self._q_net, self._loss_object, self._optimizer = self.build_q_network(q_net_conf)
        self._target_q_net, _, _ = self.build_q_network(q_net_conf)
        self._target_q_net.set_weights(self._q_net.get_weights())
        self._batch_size = batch_size
        # learning related params
        self._replay_buffer = ReplayBuffer(buffer_size, obs_dim[0])
        self._min_replay_history = min_replay_history
        self._discount_rate = discount_rate
        self._train_freq = train_freq
        self._target_network_update_freq = target_network_update_freq
        self._step_count = 0
        # epsilon greedy related params
        self._epsilon = epsilon
        self._min_epsilon = min_epsilon
        self._epsilon_decay = epsilon_decay
        self._test_epsilon = test_epsilon
        self.load_buffer = load_buffer
        # pruning attributes
        if layer_names is None:
            layer_names = ['dense', 'dense_1']
        self._pruned_q_net = None
        self._region_locator = DNNRegionsLocator(self._q_net, layer_names=layer_names, nb_observations=nb_observations,
                                                 sparsity=sparsity, epsilon=f_epsilon)

    def build_q_network(self, q_net_conf):
        # init = tf.keras.initializers.HeUniform()
        # loss_object = tf.keras.losses.Huber()
        optimizer = tf.keras.optimizers.Adam(lr=q_net_conf['lr'])
        loss_object = tf.keras.losses.MeanSquaredError()
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(q_net_conf['kernel_sizes'][0], input_dim=self.obs_dim[0], activation='relu'))
        model.add(tf.keras.layers.Dense(q_net_conf['kernel_sizes'][1], activation='relu'))
        model.add(tf.keras.layers.Dense(self.act_dim, activation=None))
        model.compile(loss='mse', optimizer=optimizer)
        return model, loss_object, optimizer

    def action(self, state, pruning=False):

        if self._training:
            epsilon = self._epsilon
        else:
            epsilon = self._test_epsilon

        if np.random.rand() < epsilon:
            # Explore
            action = np.random.randint(self.act_dim)
        else:
            # Exploit: Use the NN to predict the correct action from this state
            state = state.reshape((1, -1))
            if pruning:
                q_vals = self._pruned_q_net(state)
            else:
                # q_vals = self._q_net.predict(state)
                q_vals = self._q_net(state)
            # action = np.argmax(q_vals[0])
            action = tf.math.argmax(q_vals, axis=1).numpy()[0]

        return action

    def update(self, data, pruning=False):
        if not self._training:
            return self._test_epsilon

        # Add the most recent transition to the replay buffer.
        state, action, reward, new_state, done = data
        self._replay_buffer.add(state, action, reward, new_state, done)

        # Update the q network based on a sample batch from the replay buffer.
        # If the replay buffer doesn't have enough samples, move on.
        # if (self._replay_buffer.get_current_size() >= self._batch_size
        #         and self._replay_buffer.get_current_size() >= self._min_replay_history
        #         and self._step_count % self._train_freq == 0):

        if self._replay_buffer.get_current_size() < self._batch_size:
            return self._epsilon

        # Update target network
        if self._step_count % self._target_network_update_freq == 0:
            self._target_q_net.set_weights(self._q_net.get_weights())

        mini_batch = self._replay_buffer.sample(batch_size=self._batch_size)
        state_batch, action_batch, reward_batch, new_state_batch, done_batch = mini_batch

        # Compute predicted Q values & next Q values

        if pruning:
            q_predicted = self._pruned_q_net(state_batch)
        else:
            q_predicted = self._q_net(state_batch)

        q_next = self._target_q_net(new_state_batch)

        q_max_next = tf.math.reduce_max(q_next, axis=1, keepdims=True).numpy()
        q_target = np.copy(q_predicted)

        for idx in range(done_batch.shape[0]):
            target_q_val = reward_batch[idx]
            if not done_batch[idx]:
                target_q_val += self._discount_rate * q_max_next[idx]
            q_target[idx, action_batch[idx]] = target_q_val

        # the above few lines can be reduced and vectorized easily

        # this needs to be customized and expanded
        # self.q_net.train_on_batch(state_batch, q_target)

        if pruning:
            # with tf.GradientTape() as tape:
            #     q_pred = self._pruned_q_net(state_batch, training=True)
            #     loss_value = self._loss_object(q_target, q_pred)
            #     grads = tape.gradient(loss_value, self._pruned_q_net.trainable_variables)
            #     self._optimizer.apply_gradients(zip(grads, self._pruned_q_net.trainable_variables))
            self._pruned_q_net.train_on_batch(state_batch, q_target)
        else:
            self._region_locator.add_observations(state_batch)
            self._q_net.train_on_batch(state_batch, q_target)
            # self._q_net.fit(state_batch, q_target, batch_size=self._batch_size, verbose=0, shuffle=True)

        # self._epsilon = self._min_epsilon + (self._max_epsilon - self._min_epsilon) * np.exp(-self._epsilon_decay)
        # self._epsilon *= self._epsilon_decay

        self._epsilon = self._epsilon - self._epsilon_decay if self._epsilon > self._min_epsilon else self._min_epsilon
        self._step_count += 1

        return self._epsilon

    def save(self, dir_path):
        q_net_file_path = os.path.join(dir_path, 'q_net.h5')
        target_q_net_file_path = os.path.join(dir_path, 'target_q_net.h5')
        pruned_q_net_file_path = os.path.join(dir_path, 'pruned_q_net.h5')
        self._q_net.save(q_net_file_path)
        self._target_q_net.save(target_q_net_file_path)
        self._replay_buffer.save(dir_path)
        self._region_locator.save_observations(dir_path)
        if self._pruned_q_net is not None:
            self._pruned_q_net.save(pruned_q_net_file_path)

    def save_pruned_q_net(self, dir_path):
        pruned_q_net_file_path = os.path.join(dir_path, 'pruned_q_net.h5')
        if self._pruned_q_net is not None:
            self._pruned_q_net.save(pruned_q_net_file_path)

    def load(self, dir_path, load_observation=True):
        q_net_file_path = os.path.join(dir_path, 'q_net.h5')
        target_q_net_file_path = os.path.join(dir_path, 'target_q_net.h5')
        self._q_net = tf.keras.models.load_model(q_net_file_path)

        self._target_q_net = tf.keras.models.load_model(target_q_net_file_path)
        if self.load_buffer:
            self._replay_buffer.load(dir_path)
        if load_observation:
            self._region_locator.load_observations(dir_path)
        self._region_locator.set_model(self._q_net)
        self._region_locator.set_layer_names([self._q_net.layers[0].name, self._q_net.layers[1].name])
        # self._pruned_q_net = tf.keras.models.load_model(pruned_q_net_file_path)
        # self.compile_pruned_model()

    def load_pruned_q_net(self, dir_path):
        pruned_q_net_file_path = os.path.join(dir_path, 'pruned_q_net.h5')
        self._pruned_q_net = tf.keras.models.load_model(pruned_q_net_file_path)

    def setup_pruning(self):

        self._pruned_q_net, _, _ = self.build_q_network(self._q_net_conf)
        self._pruned_q_net.set_weights(self._q_net.get_weights())

        # Non-boilerplate.
        self._pruned_q_net.optimizer = self._optimizer

        # we need to perform pruning here
        masks = self._region_locator.generate_regions_masks()
        model_weights = self._pruned_q_net.get_weights()
        self._pruned_q_net = self._region_locator.apply_mask(masks, self._pruned_q_net)
        new_model_weights = self._pruned_q_net.get_weights()
        print()

    def compile_pruned_model(self):
        self._pruned_q_net.compile(optimizer=self._optimizer, loss=self._loss_object, metrics=['accuracy'])


if __name__ == '__main__':
    env = gym.make("Acrobot-v1")
    print()
