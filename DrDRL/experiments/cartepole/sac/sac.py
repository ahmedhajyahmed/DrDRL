"""
adapted from
https://github.com/anita-hu/TF2-RL
"""
import os
from collections import deque
import random
from tensorflow.python.keras.layers import Concatenate
from dr_drl.dnn_regions_locator import DNNRegionsLocator
from dr_drl.agents.agent import Agent
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
import tensorflow_probability as tfp
import pickle

tfd = tfp.distributions
tf.keras.backend.set_floatx('float64')


def actor(state_shape, action_shape, units=(400, 300, 100)):
    state = Input(shape=state_shape)
    x = Dense(units[0], name="L0", activation="relu")(state)
    for index in range(1, len(units)):
        x = Dense(units[index], name="L{}".format(index), activation="relu")(x)

    actions_mean = Dense(action_shape[0], name="Out_mean")(x)
    actions_std = Dense(action_shape[0], name="Out_std")(x)

    model = Model(inputs=state, outputs=[actions_mean, actions_std])

    return model


# https://github.com/anita-hu/TF2-RL
class SACAgent(Agent):
    def __init__(self, obs_dim, act_dim, act_shape, act_high_low, lr_actor=3e-5, lr_critic=3e-4, actor_units=(64, 64),
                 critic_units=(64, 64), auto_alpha=True, alpha=0.2, tau=0.005, gamma=0.99, batch_size=128,
                 memory_cap=100000, layer_names=None, nb_observations=50000, sparsity=0.5, epsilon=1, load_buffer=True):
        super().__init__(obs_dim, act_dim)

        self.action_shape = act_shape
        act_high, act_low = act_high_low
        self.action_bound = (act_high - act_low) / 2
        self.action_shift = (act_high + act_low) / 2

        self.memory = deque(maxlen=int(memory_cap))

        # Define and initialize actor network
        self.actor_units = actor_units
        self.actor = actor(self.obs_dim, self.action_shape, self.actor_units)
        self.actor_optimizer = Adam(learning_rate=lr_actor)
        self.log_std_min = -20
        self.log_std_max = 2
        # print(self.actor.summary())

        # Define and initialize critic networks
        self.critic_1 = self.critic(self.obs_dim, self.action_shape, critic_units)
        self.critic_target_1 = self.critic(self.obs_dim, self.action_shape, critic_units)
        self.critic_optimizer_1 = Adam(learning_rate=lr_critic)
        self.update_target_weights(self.critic_1, self.critic_target_1, tau=1.)

        self.critic_2 = self.critic(self.obs_dim, self.action_shape, critic_units)
        self.critic_target_2 = self.critic(self.obs_dim, self.action_shape, critic_units)
        self.critic_optimizer_2 = Adam(learning_rate=lr_critic)
        self.update_target_weights(self.critic_2, self.critic_target_2, tau=1.)

        # Define and initialize temperature alpha and target entropy
        self.auto_alpha = auto_alpha
        if auto_alpha:
            self.target_entropy = -np.prod(self.action_shape)
            self.log_alpha = tf.Variable(0., dtype=tf.float64)
            self.alpha = tf.Variable(0., dtype=tf.float64)
            self.alpha.assign(tf.exp(self.log_alpha))
            self.alpha_optimizer = Adam(learning_rate=lr_actor)
        else:
            self.alpha = tf.Variable(alpha, dtype=tf.float64)

        # Set hyper parameters
        self.gamma = gamma  # discount factor
        self.tau = tau  # target model update
        self.batch_size = batch_size

        # pruning attributes
        if layer_names is None:
            layer_names = ['L0', 'L1', 'L2']
        self._pruned_actor = None
        self._region_locator = DNNRegionsLocator(self.actor, layer_names=layer_names, nb_observations=nb_observations,
                                                 sparsity=sparsity, epsilon=epsilon)
        self.load_buffer= load_buffer

    def critic(self, state_shape, action_shape, units=(400, 200, 100)):
        inputs = [Input(shape=state_shape), Input(shape=action_shape)]
        concat = Concatenate(axis=-1)(inputs)
        x = Dense(units[0], name="Hidden0", activation="relu")(concat)
        for index in range(1, len(units)):
            x = Dense(units[index], name="Hidden{}".format(index), activation="relu")(x)

        output = Dense(1, name="Out_QVal")(x)
        model = Model(inputs=inputs, outputs=output)

        return model

    def update_target_weights(self, model, target_model, tau=0.005):
        if not self._training:
            return
        weights = model.get_weights()
        target_weights = target_model.get_weights()
        for i in range(len(target_weights)):  # set tau% of target model to be new weights
            target_weights[i] = weights[i] * tau + target_weights[i] * (1 - tau)
        target_model.set_weights(target_weights)

    def process_actions(self, mean, log_std, test=False, eps=1e-6):
        std = tf.dtypes.cast(tf.math.exp(log_std), dtype=tf.float64)
        raw_actions = mean

        if not test:
            raw_actions += tf.math.multiply(tf.random.normal(shape=mean.shape, dtype=tf.float64), std)

        log_prob_u = tfd.Normal(loc=mean, scale=std).log_prob(raw_actions)
        actions = tf.math.tanh(raw_actions)

        log_prob = tf.reduce_sum(log_prob_u - tf.math.log(1 - actions ** 2 + eps))

        actions = actions * self.action_bound + self.action_shift

        return actions, log_prob

    def action(self, state, use_random=False, pruning=False):
        state = np.expand_dims(state, axis=0).astype(np.float64)

        if use_random:
            a = tf.random.uniform(shape=(1, self.action_shape[0]), minval=-1, maxval=1, dtype=tf.float64)
        else:
            if pruning:
                means, log_stds = self._pruned_actor.predict(state)
            else:
                means, log_stds = self.actor.predict(state)
            log_stds = tf.clip_by_value(log_stds, self.log_std_min, self.log_std_max)

            a, log_prob = self.process_actions(means, log_stds, test=(not self._training))

        return a

    def remember(self, state, action, reward, next_state, done):
        state = np.expand_dims(state, axis=0)
        next_state = np.expand_dims(next_state, axis=0)
        self.memory.append([state, action, reward, next_state, done])

    def update(self, data=None, pruning=False):
        if not self._training:
            return
        if len(self.memory) < self.batch_size:
            return

        samples = random.sample(self.memory, self.batch_size)
        s = np.array(samples).T
        states, actions, rewards, next_states, dones = [np.vstack(s[i, :]).astype(np.float) for i in range(5)]

        if not pruning:
            # collect observation to determine major/minor regions (for 'forget mechanism')
            self._region_locator.add_observations(next_states)

        with tf.GradientTape(persistent=True) as tape:
            # next state action log probs
            if pruning:
                means, log_stds = self._pruned_actor(next_states)
            else:
                means, log_stds = self.actor(next_states)
            log_stds = tf.clip_by_value(log_stds, self.log_std_min, self.log_std_max)
            next_actions, log_probs = self.process_actions(means, log_stds)

            # critics loss
            current_q_1 = self.critic_1([states, actions])
            current_q_2 = self.critic_2([states, actions])
            next_q_1 = self.critic_target_1([next_states, next_actions])
            next_q_2 = self.critic_target_2([next_states, next_actions])
            next_q_min = tf.math.minimum(next_q_1, next_q_2)
            state_values = next_q_min - self.alpha * log_probs
            target_qs = tf.stop_gradient(rewards + state_values * self.gamma * (1. - dones))
            critic_loss_1 = tf.reduce_mean(0.5 * tf.math.square(current_q_1 - target_qs))
            critic_loss_2 = tf.reduce_mean(0.5 * tf.math.square(current_q_2 - target_qs))

            # current state action log probs
            if pruning:
                means, log_stds = self._pruned_actor(states)
            else:
                means, log_stds = self.actor(states)
            log_stds = tf.clip_by_value(log_stds, self.log_std_min, self.log_std_max)
            actions, log_probs = self.process_actions(means, log_stds)

            # actor loss
            current_q_1 = self.critic_1([states, actions])
            current_q_2 = self.critic_2([states, actions])
            current_q_min = tf.math.minimum(current_q_1, current_q_2)
            actor_loss = tf.reduce_mean(self.alpha * log_probs - current_q_min)

            # temperature loss
            if self.auto_alpha:
                alpha_loss = -tf.reduce_mean(
                    (self.log_alpha * tf.stop_gradient(log_probs + self.target_entropy)))

        critic_grad = tape.gradient(critic_loss_1, self.critic_1.trainable_variables)  # compute actor gradient
        self.critic_optimizer_1.apply_gradients(zip(critic_grad, self.critic_1.trainable_variables))

        critic_grad = tape.gradient(critic_loss_2, self.critic_2.trainable_variables)  # compute actor gradient
        self.critic_optimizer_2.apply_gradients(zip(critic_grad, self.critic_2.trainable_variables))

        if pruning:
            trainable_variables = self._pruned_actor.trainable_variables
        else:
            trainable_variables = self.actor.trainable_variables

        actor_grad = tape.gradient(actor_loss, trainable_variables)  # compute actor gradient
        self.actor_optimizer.apply_gradients(zip(actor_grad, trainable_variables))

        if self.auto_alpha:
            # optimize temperature
            alpha_grad = tape.gradient(alpha_loss, [self.log_alpha])
            self.alpha_optimizer.apply_gradients(zip(alpha_grad, [self.log_alpha]))
            self.alpha.assign(tf.exp(self.log_alpha))

    def save(self, dir_path):
        actor_file_path = os.path.join(dir_path, 'actor.h5')
        critic1_file_path = os.path.join(dir_path, 'critic1.h5')
        critic2_file_path = os.path.join(dir_path, 'critic2.h5')
        pruned_actor_file_path = os.path.join(dir_path, 'pruned_actor.h5')
        replay_buffer_file_path = os.path.join(dir_path, 'replay_buffer_pickle')
        self.actor.save(actor_file_path)
        self.critic_1.save(critic1_file_path)
        self.critic_2.save(critic2_file_path)
        file = open(replay_buffer_file_path, 'ab')
        pickle.dump(self.memory, file)
        file.close()
        self._region_locator.save_observations(dir_path)
        if self._pruned_actor is not None:
            self._pruned_actor.save(pruned_actor_file_path)

    def save_pruned_actor(self, dir_path):
        pruned_actor_file_path = os.path.join(dir_path, 'pruned_actor.h5')
        if self._pruned_actor is not None:
            self._pruned_actor.save(pruned_actor_file_path)

    def load(self, dir_path, load_observation=True):
        actor_file_path = os.path.join(dir_path, 'actor.h5')
        critic1_file_path = os.path.join(dir_path, 'critic1.h5')
        critic2_file_path = os.path.join(dir_path, 'critic2.h5')
        replay_buffer_file_path = os.path.join(dir_path, 'replay_buffer_pickle')
        self.actor = tf.keras.models.load_model(actor_file_path)
        self.critic_1 = tf.keras.models.load_model(critic1_file_path)
        self.critic_2 = tf.keras.models.load_model(critic2_file_path)
        if self.load_buffer:
            file = open(replay_buffer_file_path, 'rb')
            self.memory = pickle.load(file)
            file.close()
        if load_observation:
            self._region_locator.load_observations(dir_path)
        self._region_locator.set_layer_names([self.actor.layers[1].name, self.actor.layers[2].name])

    def load_pruned_actor(self, dir_path):
        pruned_actor_file_path = os.path.join(dir_path, 'pruned_actor.h5')
        self._pruned_actor = tf.keras.models.load_model(pruned_actor_file_path)

    def setup_pruning(self):
        self._pruned_actor = actor(self.obs_dim, self.action_shape, self.actor_units)
        self._pruned_actor.set_weights(self.actor.get_weights())

        # Non-boilerplate.
        self._pruned_actor.optimizer = self.actor_optimizer

        # we need to perform pruning here
        masks = self._region_locator.generate_regions_masks()
        model_weights = self._pruned_actor.get_weights()
        self._pruned_actor = self._region_locator.apply_mask(masks, self._pruned_actor)
        new_model_weights = self._pruned_actor.get_weights()
        print()
