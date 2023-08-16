"""
This code was udapted from
https://github.com/anita-hu/TF2-RL
"""
import copy
import os
from dr_drl.agents.agent import Agent
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
import tensorflow_probability as tfp
from dr_drl.dnn_regions_locator import DNNRegionsLocator

tfd = tfp.distributions
tf.keras.backend.set_floatx('float64')


# https://github.com/anita-hu/TF2-RL
class PPOAgent(Agent):
    def __init__(self, obs_dim, act_dim, act_high_low, policy_conf, batch_size=64, discrete=False, c1=1.0, c2=0.01,
                 clip_ratio=0.2, gamma=0.95, lam=1.0, n_updates=4, layer_names=None, nb_observations=50000,
                 sparsity=0.5, epsilon=1):
        super().__init__(obs_dim, act_dim)
        if not discrete:
            action_space_high, action_space_low = act_high_low

        self.discrete = discrete
        # DNN related params
        self.policy_conf = policy_conf
        self.policy = self.build_model(policy_conf)
        self.model_optimizer = Adam(learning_rate=policy_conf['lr'])
        self._batch_size = batch_size

        if not discrete:
            self.action_bound = (action_space_high - action_space_low) / 2
            self.action_shift = (action_space_high + action_space_low) / 2

        # Stdev for continuous action
        if not discrete:
            self.policy_log_std = tf.Variable(tf.zeros(self.act_dim, dtype=tf.float64), trainable=True)

        # Set hyperparameters
        self.gamma = gamma  # discount factor
        self.lam = lam
        self.c1 = c1  # value difference coeff
        self.c2 = c2  # entropy coeff
        self.clip_ratio = clip_ratio  # for clipped surrogate
        self.batch_size = batch_size
        self.n_updates = n_updates  # number of epochs per episode
        # pruning attributes
        if layer_names is None:
            layer_names = ['Policy_L0', 'Policy_L1']
        self._pruned_policy = None
        self._region_locator = DNNRegionsLocator(self.policy, layer_names=layer_names, nb_observations=nb_observations,
                                                 sparsity=sparsity, epsilon=epsilon)

    def build_model(self, q_net_conf):
        state = Input(shape=self.obs_dim)

        vf = Dense(q_net_conf["units"][0], name="Value_L0", activation="tanh")(state)
        for index in range(1, len(q_net_conf["units"])):
            vf = Dense(q_net_conf["units"][index], name="Value_L{}".format(index), activation="tanh")(vf)

        value_pred = Dense(1, name="Out_value")(vf)

        pi = Dense(q_net_conf["units"][0], name="Policy_L0", activation="tanh")(state)
        for index in range(1, len(q_net_conf["units"])):
            pi = Dense(q_net_conf["units"][index], name="Policy_L{}".format(index), activation="tanh")(pi)

        if self.discrete:
            action_probs = Dense(self.act_dim, name="Out_probs", activation='softmax')(pi)
            model = Model(inputs=state, outputs=[action_probs, value_pred])
        else:
            actions_mean = Dense(self.act_dim, name="Out_mean", activation='tanh')(pi)
            model = Model(inputs=state, outputs=[actions_mean, value_pred])

        # loss_object

        return model

    def action(self, state, pruning=False):
        state = np.expand_dims(state, axis=0).astype(np.float64)
        if pruning:
            output, value = self._pruned_policy.predict(state)
        else:
            output, value = self.policy.predict(state)
        dist = self.get_dist(output)

        if self.discrete:
            action = tf.math.argmax(output, axis=-1) if not (self._training) else dist.sample()
            log_probs = dist.log_prob(action)
        else:
            action = output if not (self._training) else dist.sample()
            action = tf.clip_by_value(action, -1, 1)
            log_probs = tf.reduce_sum(dist.log_prob(action), axis=-1)
            action = action * self.action_bound + self.action_shift

        return action[0].numpy(), value[0][0], log_probs[0].numpy()

    def get_gaes(self, rewards, v_preds, next_v_preds):
        # source: https://github.com/uidilr/ppo_tf/blob/master/ppo.py#L98
        deltas = [r_t + self.gamma * v_next - v for r_t, v_next, v in zip(rewards, next_v_preds, v_preds)]
        gaes = copy.deepcopy(deltas)
        for t in reversed(range(len(gaes) - 1)):  # is T-1, where T is time step which run policy
            gaes[t] = gaes[t] + self.lam * self.gamma * gaes[t + 1]
        return gaes

    def get_dist(self, output):
        if self.discrete:
            dist = tfd.Categorical(probs=output)
        else:
            std = tf.math.exp(self.policy_log_std)
            dist = tfd.Normal(loc=output, scale=std)

        return dist

    def evaluate_actions(self, state, action, pruning=False):
        if pruning:
            output, value = self._pruned_policy(state)
        else:
            output, value = self.policy(state)
        dist = self.get_dist(output)
        if not self.discrete:
            action = (action - self.action_shift) / self.action_bound

        log_probs = dist.log_prob(action)
        if not self.discrete:
            log_probs = tf.reduce_sum(log_probs, axis=-1)

        entropy = dist.entropy()

        return log_probs, entropy, value

    def update(self, data, pruning=False):
        observations, actions, log_probs, next_v_preds, rewards, gaes = data

        rewards = np.expand_dims(rewards, axis=-1).astype(np.float64)
        next_v_preds = np.expand_dims(next_v_preds, axis=-1).astype(np.float64)

        if not pruning:
            # collect observation to determine major/minor regions (for 'forget mechanism')
            self._region_locator.add_observations(observations)
        with tf.GradientTape() as tape:
            new_log_probs, entropy, state_values = self.evaluate_actions(observations, actions, pruning=pruning)

            ratios = tf.exp(new_log_probs - log_probs)
            clipped_ratios = tf.clip_by_value(ratios, clip_value_min=1 - self.clip_ratio,
                                              clip_value_max=1 + self.clip_ratio)
            loss_clip = tf.minimum(gaes * ratios, gaes * clipped_ratios)
            loss_clip = tf.reduce_mean(loss_clip)

            target_values = rewards + self.gamma * next_v_preds
            vf_loss = tf.reduce_mean(tf.math.square(state_values - target_values))

            entropy = tf.reduce_mean(entropy)
            total_loss = -loss_clip + self.c1 * vf_loss - self.c2 * entropy

        if pruning:
            train_variables = self._pruned_policy.trainable_variables
        else:
            train_variables = self.policy.trainable_variables
        if not self.discrete:
            train_variables += [self.policy_log_std]
        grad = tape.gradient(total_loss, train_variables)  # compute gradient
        self.model_optimizer.apply_gradients(zip(grad, train_variables))

    def save(self, dir_path):
        policy_path = os.path.join(dir_path, 'policy.h5')
        pruned_policy_file_path = os.path.join(dir_path, 'pruned_policy.h5')
        self.policy.save(policy_path)
        self._region_locator.save_observations(dir_path)
        if self._pruned_policy is not None:
            self._pruned_policy.save(pruned_policy_file_path)

    def save_pruned_policy(self, dir_path):
        pruned_policy_file_path = os.path.join(dir_path, 'pruned_policy.h5')
        if self._pruned_policy is not None:
            self._pruned_policy.save(pruned_policy_file_path)

    def load(self, dir_path, load_observation=True):
        policy_path = os.path.join(dir_path, 'policy.h5')
        self.policy = tf.keras.models.load_model(policy_path)
        if load_observation:
            self._region_locator.load_observations(dir_path)
        self._region_locator.set_model(self.policy)
        self._region_locator.set_layer_names([self.policy.layers[1].name, self.policy.layers[3].name])

    def load_pruned_policy(self, dir_path):
        pruned_policy_file_path = os.path.join(dir_path, 'pruned_policy.h5')
        self._pruned_policy = tf.keras.models.load_model(pruned_policy_file_path)

    def setup_pruning(self):

        self._pruned_policy = self.build_model(self.policy_conf)
        self._pruned_policy.set_weights(self.policy.get_weights())

        # Non-boilerplate.
        self._pruned_policy.optimizer = self.model_optimizer

        # we need to perform pruning here
        masks = self._region_locator.generate_regions_masks()
        model_weights = self._pruned_policy.get_weights()
        self._pruned_policy = self._region_locator.apply_mask(masks, self._pruned_policy)
        new_model_weights = self._pruned_policy.get_weights()
        print()
