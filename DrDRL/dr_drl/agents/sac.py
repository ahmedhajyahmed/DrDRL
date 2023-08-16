"""
This code was udapted from
https://github.com/anita-hu/TF2-RL
"""
import os
from collections import deque
import random
from dr_drl.dnn_regions_locator import DNNRegionsLocator
from dr_drl.agents.agent import Agent
import tensorflow as tf
import numpy as np
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Dense
import tensorflow_probability as tfp
import pickle
from typing import Sequence
import tensorflow_addons as tfa


class ReplayBuffer(object):
    def __init__(self, buffer_size: int) -> None:
        self.buffer_size: int = buffer_size
        self.num_experiences: int = 0
        self.buffer: deque = deque()

    def get_batch(self, batch_size: int) -> dict:
        # Randomly sample batch_size examples
        experiences = random.sample(self.buffer, batch_size)
        return {
            "states0": np.asarray([exp[0] for exp in experiences], np.float32),
            "actions": np.asarray([exp[1] for exp in experiences], np.float32),
            "rewards": np.asarray([exp[2] for exp in experiences], np.float32),
            "states1": np.asarray([exp[3] for exp in experiences], np.float32),
            "terminals1": np.asarray([exp[4] for exp in experiences], np.float32)
        }

    def add(self, state: np.ndarray, action: np.ndarray, reward: float, new_state: np.ndarray, done: bool):
        experience = (state, action, reward, new_state, done)
        if self.num_experiences < self.buffer_size:
            self.buffer.append(experience)
            self.num_experiences += 1
        else:
            self.buffer.popleft()
            self.buffer.append(experience)

    @property
    def size(self) -> int:
        return self.buffer_size

    @property
    def n_entries(self) -> int:
        # If buffer is full, return buffer size
        # Otherwise, return experience counter
        return self.num_experiences

    def erase(self):
        self.buffer = deque()
        self.num_experiences = 0

    def save(self, dir_path):
        replay_buffer_file_path = os.path.join(dir_path, 'replay_buffer.pickle')
        replay_buffer_counter_file_path = os.path.join(dir_path, 'replay_buffer_counter.npy')
        file = open(replay_buffer_file_path, 'ab')
        pickle.dump(self.buffer, file)
        file.close()
        np.save(replay_buffer_counter_file_path, np.array(self.num_experiences))

    def load(self, dir_path):
        replay_buffer_file_path = os.path.join(dir_path, 'replay_buffer.pickle')
        replay_buffer_counter_file_path = os.path.join(dir_path, 'replay_buffer_counter.npy')
        file = open(replay_buffer_file_path, 'rb')
        self.buffer = pickle.load(file)
        file.close()
        self.num_experiences = int(np.load(replay_buffer_counter_file_path))


class ActorNetwork(Model):
    def __init__(self, n_hidden_layers, n_hidden_units, n_actions, logprob_epsilon):
        super(ActorNetwork, self).__init__()
        self.logprob_epsilon = logprob_epsilon
        w_bound = 3e-3
        self.hidden = Sequential()
        for _ in range(n_hidden_layers):
            self.hidden.add(Dense(n_hidden_units, activation="relu"))

        self.mean = Dense(n_actions,
                          kernel_initializer=tf.random_uniform_initializer(-w_bound, w_bound),
                          bias_initializer=tf.random_uniform_initializer(-w_bound, w_bound))
        self.log_std = Dense(n_actions,
                             kernel_initializer=tf.random_uniform_initializer(-w_bound, w_bound),
                             bias_initializer=tf.random_uniform_initializer(-w_bound, w_bound))

    @tf.function
    def call(self, inp):
        x = self.hidden(inp)
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std_clipped = tf.clip_by_value(log_std, -20, 2)
        normal_dist = tfp.distributions.Normal(mean, tf.exp(log_std_clipped))
        action = tf.stop_gradient(normal_dist.sample())
        squashed_actions = tf.tanh(action)
        logprob = normal_dist.log_prob(action) - tf.math.log(1.0 - tf.pow(squashed_actions, 2) + self.logprob_epsilon)
        logprob = tf.reduce_sum(logprob, axis=-1, keepdims=True)
        return squashed_actions, logprob

    def _get_params(self):
        ''
        with self.graph.as_default():
            params = tf.trainable_variables()
        names = [p.name for p in params]
        values = self.sess.run(params)
        params = {k: v for k, v in zip(names, values)}
        return params

    def __getstate__(self):
        params = self._get_params()
        state = self.args_copy, params
        return state

    def __setstate__(self, state):
        args, params = state
        self.__init__(**args)
        self.restore_params(params)


class SoftQNetwork(Model):
    def __init__(self, n_hidden_layers, n_hidden_units):
        super(SoftQNetwork, self).__init__()
        self.softq = Sequential()
        for _ in range(n_hidden_layers):
            self.softq.add(Dense(n_hidden_units, activation="relu"))
        self.softq.add(Dense(1,
                             kernel_initializer=tf.random_uniform_initializer(-3e-3, 3e-3),
                             bias_initializer=tf.random_uniform_initializer(-3e-3, 3e-3)))

    @tf.function
    def call(self, states, actions):
        x = tf.concat([states, actions], 1)
        return self.softq(x)


class ValueNetwork(Model):
    def __init__(self, n_hidden_layers, n_hidden_units):
        super(ValueNetwork, self).__init__()
        self.value = Sequential()
        for _ in range(n_hidden_layers):
            self.value.add(Dense(n_hidden_units, activation="relu"))

        self.value.add(Dense(1,
                             kernel_initializer=tf.random_uniform_initializer(-3e-3, 3e-3),
                             bias_initializer=tf.random_uniform_initializer(-3e-3, 3e-3)))

    def call(self, inp):
        return self.value(inp)


def soft_update(source_vars: Sequence[tf.Variable], target_vars: Sequence[tf.Variable], tau: float) -> None:
    """Move each source variable by a factor of tau towards the corresponding target variable.
    Arguments:
        source_vars {Sequence[tf.Variable]} -- Source variables to copy from
        target_vars {Sequence[tf.Variable]} -- Variables to copy data to
        tau {float} -- How much to change to source var, between 0 and 1.
    """
    if len(source_vars) != len(target_vars):
        raise ValueError("source_vars and target_vars must have the same length.")
    for source, target in zip(source_vars, target_vars):
        target.assign((1.0 - tau) * target + tau * source)


def hard_update(source_vars: Sequence[tf.Variable], target_vars: Sequence[tf.Variable]) -> None:
    """Copy source variables to target variables.
    Arguments:
        source_vars {Sequence[tf.Variable]} -- Source variables to copy from
        target_vars {Sequence[tf.Variable]} -- Variables to copy data to
    """
    # Tau of 1, so get everything from source and keep nothing from target
    soft_update(source_vars, target_vars, 1.0)


# https://github.com/RickyMexx/SAC-tf2
class SACAgent(Agent):
    def __init__(self, obs_dim, act_dim, act_lim, discount=0.99, temperature=0.3, polyak_coef=0.01, lr=1e-3,
                 hidden_layers=2, n_hidden_units=256, batch_size=128, replay_start_size=50000, buffer_size=int(1e6),
                 layer_names=None, nb_observations=50000, sparsity=0.5, epsilon=1, load_buffer=True):
        super().__init__(obs_dim, act_dim)

        self.n_actions = act_dim
        self.act_lim = act_lim
        self.discount = discount
        self.temperature = temperature
        self.polyak_coef = polyak_coef
        self.lr = lr
        self.gamma = discount
        self.replay_start_size = replay_start_size
        self.batch_size = batch_size

        # Creating a ReplayBuffer for the training process
        self.replay_buffer = ReplayBuffer(buffer_size)

        # Creating networks and optimizers ###
        # Policy network
        # action_output are the squashed actions and action_original those straight from the normal distribution
        self.logprob_epsilon = 1e-6  # For numerical stability when computing tf.log
        self.hidden_layers = hidden_layers
        self.n_hidden_units = n_hidden_units
        self.actor_network = ActorNetwork(hidden_layers, n_hidden_units, self.n_actions, self.logprob_epsilon)

        # 2 Soft q-functions networks + targets
        self.softq_network = SoftQNetwork(hidden_layers, n_hidden_units)
        self.softq_target_network = SoftQNetwork(hidden_layers, n_hidden_units)

        self.softq_network2 = SoftQNetwork(hidden_layers, n_hidden_units)
        self.softq_target_network2 = SoftQNetwork(hidden_layers, n_hidden_units)

        # Building up 2 soft q-function with their relative targets
        input1 = tf.keras.Input(shape=(obs_dim), dtype=tf.float32)
        input2 = tf.keras.Input(shape=(self.n_actions), dtype=tf.float32)

        self.softq_network(input1, input2)
        self.softq_target_network(input1, input2)
        hard_update(self.softq_network.variables, self.softq_target_network.variables)

        self.softq_network2(input1, input2)
        self.softq_target_network2(input1, input2)
        hard_update(self.softq_network2.variables, self.softq_target_network2.variables)

        # Optimizers for the networks
        self.softq_optimizer = tfa.optimizers.RectifiedAdam(learning_rate=lr)
        self.softq_optimizer2 = tfa.optimizers.RectifiedAdam(learning_rate=lr)
        self.actor_optimizer = tfa.optimizers.RectifiedAdam(learning_rate=lr)

        # pruning attributes
        if layer_names is None:
            layer_names = ['L0', 'L1']
        self._pruned_actor_network = None
        self._region_locator = DNNRegionsLocator(self.actor_network.hidden, layer_names=layer_names,
                                                 nb_observations=nb_observations,
                                                 sparsity=sparsity, epsilon=epsilon)
        self.load_buffer = load_buffer

    def softq_value(self, states: np.ndarray, actions: np.ndarray):
        return self.softq_network(states, actions)

    def softq_value2(self, states: np.ndarray, actions: np.ndarray):
        return self.softq_network2(states, actions)

    def actions(self, states: np.ndarray) -> np.ndarray:
        """Get the actions for a batch of states."""
        return self.actor_network(states)[0]

    def action(self, state: np.ndarray, pruning=False) -> np.ndarray:
        """Get the action for a single state."""

        if pruning:
            action = self._pruned_actor_network(state[None, :])[0][0]
        else:
            action = self.actor_network(state[None, :])[0][0]

        return action

    def step(self, obs):
        return self.actor_network(obs)[0]

    def update(self, check, pruning=False):
        if not self._training:
            return check

        if self.replay_buffer.n_entries > self.replay_start_size:
            if check == 1:
                print("The buffer is ready, training is starting!")
                check = 0

            sample = self.replay_buffer.get_batch(self.batch_size)
            if not pruning:
                # collect observation to determine major/minor regions (for 'forget mechanism')
                self._region_locator.add_observations(sample["states0"])
            train_metrics = self.train(sample, np.resize(sample["actions"], [self.batch_size, self.n_actions]),
                                       self.batch_size, pruning=pruning)

        return check

    @tf.function
    def train(self, sample, action_batch, batch_size, pruning=False):
        state0_batch = sample["states0"]
        reward_batch = sample["rewards"]
        state1_batch = sample["states1"]
        terminal1_batch = sample["terminals1"]

        # Computing action and a_tilde
        if pruning:
            action, action_logprob2 = self._pruned_actor_network(state1_batch)
        else:
            action, action_logprob2 = self.actor_network(state1_batch)

        value_target1 = self.softq_target_network(state1_batch, action)
        value_target2 = self.softq_target_network2(state1_batch, action)

        # Taking the minimum of the q-functions values
        next_value_batch = tf.math.minimum(value_target1, value_target2) - self.temperature * action_logprob2

        # Computing target for q-functions
        softq_targets = reward_batch + self.gamma * (1 - terminal1_batch) * tf.reshape(next_value_batch, [-1])
        softq_targets = tf.reshape(softq_targets, [batch_size, 1])

        # Gradient descent for the first q-function
        with tf.GradientTape() as softq_tape:
            softq = self.softq_network(state0_batch, action_batch)
            softq_loss = tf.reduce_mean(tf.square(softq - softq_targets))

        # Gradient descent for the second q-function
        with tf.GradientTape() as softq_tape2:
            softq2 = self.softq_network2(state0_batch, action_batch)
            softq_loss2 = tf.reduce_mean(tf.square(softq2 - softq_targets))

        # Gradient ascent for the policy (actor)
        with tf.GradientTape() as actor_tape:
            if pruning:
                actions, action_logprob = self._pruned_actor_network(state0_batch)
            else:
                actions, action_logprob = self.actor_network(state0_batch)

            new_softq = tf.math.minimum(self.softq_network(state0_batch, actions),
                                        self.softq_network2(state0_batch, actions))

            # Loss implementation from the pseudocode -> works worse
            # actor_loss = tf.reduce_mean(action_logprob - new_softq)

            # New actor_loss -> works better
            advantage = tf.stop_gradient(action_logprob - new_softq)
            actor_loss = tf.reduce_mean(action_logprob * advantage)

        # Computing the gradients with the tapes and applying them
        if pruning:
            actor_gradients = actor_tape.gradient(actor_loss, self._pruned_actor_network.trainable_weights)
        else:
            actor_gradients = actor_tape.gradient(actor_loss, self.actor_network.trainable_weights)
        softq_gradients = softq_tape.gradient(softq_loss, self.softq_network.trainable_weights)
        softq_gradients2 = softq_tape2.gradient(softq_loss2, self.softq_network2.trainable_weights)

        # Minimize gradients wrt weights
        if pruning:
            self.actor_optimizer.apply_gradients(zip(actor_gradients, self._pruned_actor_network.trainable_weights))
        else:
            self.actor_optimizer.apply_gradients(zip(actor_gradients, self.actor_network.trainable_weights))
        self.softq_optimizer.apply_gradients(zip(softq_gradients, self.softq_network.trainable_weights))
        self.softq_optimizer2.apply_gradients(zip(softq_gradients2, self.softq_network2.trainable_weights))

        # Update the weights of the soft q-function target networks
        soft_update(self.softq_network.variables, self.softq_target_network.variables, self.polyak_coef)
        soft_update(self.softq_network2.variables, self.softq_target_network2.variables, self.polyak_coef)

        # Computing mean and variance of soft-q function
        softq_mean, softq_variance = tf.nn.moments(softq, axes=[0])

        return softq_mean[0], tf.sqrt(softq_variance[0]), softq_loss, actor_loss, tf.reduce_mean(action_logprob)

    def save(self, dir_path):
        actor_file_path = os.path.join(dir_path, 'actor.ckpt')
        critic1_file_path = os.path.join(dir_path, 'critic1.ckpt')
        critic2_file_path = os.path.join(dir_path, 'critic2.ckpt')
        pruned_actor_file_path = os.path.join(dir_path, 'pruned_actor.ckpt')
        self.actor_network.save_weights(actor_file_path)
        self.softq_network.save_weights(critic1_file_path)
        self.softq_network2.save_weights(critic2_file_path)
        self.replay_buffer.save(dir_path)
        self._region_locator.save_observations(dir_path)
        if self._pruned_actor_network is not None:
            self._pruned_actor_network.save_weights(pruned_actor_file_path)

    def save_pruned_actor(self, dir_path):
        pruned_actor_file_path = os.path.join(dir_path, 'pruned_actor.ckpt')
        if self._pruned_actor_network is not None:
            self._pruned_actor_network.save_weights(pruned_actor_file_path)

    def load(self, dir_path, load_observation=True, load_pruned=False):
        actor_file_path = os.path.join(dir_path, 'actor.ckpt')
        critic1_file_path = os.path.join(dir_path, 'critic1.ckpt')
        critic2_file_path = os.path.join(dir_path, 'critic2.ckpt')
        self.actor_network.load_weights(actor_file_path)
        self.softq_network.load_weights(critic1_file_path)
        self.softq_network2.load_weights(critic2_file_path)

        input1 = tf.keras.Input(shape=(self.obs_dim), dtype=tf.float32)
        input2 = tf.keras.Input(shape=(self.n_actions), dtype=tf.float32)

        self.softq_network(input1, input2)
        self.softq_target_network(input1, input2)
        hard_update(self.softq_network.variables, self.softq_target_network.variables)

        self.softq_network2(input1, input2)
        self.softq_target_network2(input1, input2)
        hard_update(self.softq_network2.variables, self.softq_target_network2.variables)

        if load_pruned:
            self._pruned_actor_network = ActorNetwork(self.hidden_layers, self.n_hidden_units, self.n_actions,
                                                      self.logprob_epsilon)
            self._pruned_actor_network.load_weights(actor_file_path)

        if self.load_buffer:
            self.replay_buffer.load(dir_path)
        if load_observation:
            self._region_locator.load_observations(dir_path)
        self._region_locator.set_layer_names([self.actor_network.layers[0].layers[0].name,
                                              self.actor_network.layers[0].layers[1].name])

    def load_pruned_actor(self, dir_path):
        pruned_actor_file_path = os.path.join(dir_path, 'pruned_actor.ckpt')
        self._pruned_actor_network.load_weights(pruned_actor_file_path)

    def setup_pruning(self):
        # self._pruned_actor_network = ActorNetwork(self.hidden_layers, self.n_hidden_units, self.n_actions,
        #                                           self.logprob_epsilon)
        # self._pruned_actor_network.set_weights(self.actor_network.get_weights())

        input1 = tf.keras.Input(shape=(self.obs_dim), dtype=tf.float32)

        self._pruned_actor_network(input1)

        # Non-boilerplate.
        self._pruned_actor_network.optimizer = self.actor_optimizer

        # we need to perform pruning here
        masks = self._region_locator.generate_regions_masks()
        model_weights = self._pruned_actor_network.get_weights()
        self._pruned_actor_network.hidden = self._region_locator.apply_mask(masks, self._pruned_actor_network.hidden)
        new_model_weights = self._pruned_actor_network.get_weights()
        print()
