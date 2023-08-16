
from dr_drl.agents.agent import Agent
from dr_drl.dnn_regions_locator import DNNRegionsLocator
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.layers import Dense
from tensorflow.keras import Model
import os
import numpy as np


class Utils:
    def prepro(self, I):
        I = I[35:195]  # crop
        I = I[::2, ::2, 0]  # downsample by factor of 2
        I[I == 144] = 0  # erase background (background type 1)
        I[I == 109] = 0  # erase background (background type 2)
        I[I != 0] = 1  # everything else (paddles, ball) just set to 1
        X = I.astype(np.float32).ravel()  # Combine items in 1 array
        return X


class Actor_Model(Model):
    def __init__(self, state_dim, action_dim):
        super(Actor_Model, self).__init__()
        self.d1 = Dense(256, activation='relu')
        self.d2 = Dense(256, activation='relu')
        self.dout = Dense(action_dim, activation='softmax')

    def call(self, x):
        x = self.d1(x)
        x = self.d2(x)
        return self.dout(x)


class Critic_Model(Model):
    def __init__(self, state_dim, action_dim):
        super(Critic_Model, self).__init__()
        self.d1 = Dense(256, activation='relu')
        self.d2 = Dense(256, activation='relu')
        self.dout = Dense(1, activation='linear')

    def call(self, x):
        x = self.d1(x)
        x = self.d2(x)
        return self.dout(x)


class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.rewards = []
        self.dones = []
        self.next_states = []

    def __len__(self):
        return len(self.dones)

    def get_all_items(self):
        states = tf.constant(self.states, dtype=tf.float32)
        actions = tf.constant(self.actions, dtype=tf.float32)
        rewards = tf.expand_dims(tf.constant(self.rewards, dtype=tf.float32), 1)
        dones = tf.expand_dims(tf.constant(self.dones, dtype=tf.float32), 1)
        next_states = tf.constant(self.next_states, dtype=tf.float32)

        return tf.data.Dataset.from_tensor_slices((states, actions, rewards, dones, next_states))

    def save_eps(self, state, action, reward, done, next_state):
        self.rewards.append(reward)
        self.states.append(state)
        self.actions.append(action)
        self.dones.append(done)
        self.next_states.append(next_state)

    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.rewards[:]
        del self.dones[:]
        del self.next_states[:]


class Distributions:
    def sample(self, datas):
        distribution = tfp.distributions.Categorical(probs=datas)
        return distribution.sample()

    def entropy(self, datas):
        distribution = tfp.distributions.Categorical(probs=datas)
        return distribution.entropy()

    def logprob(self, datas, value_data):
        distribution = tfp.distributions.Categorical(probs=datas)
        return tf.expand_dims(distribution.log_prob(value_data), 1)

    def kl_divergence(self, datas1, datas2):
        distribution1 = tfp.distributions.Categorical(probs=datas1)
        distribution2 = tfp.distributions.Categorical(probs=datas2)

        return tf.expand_dims(tfp.distributions.kl_divergence(distribution1, distribution2), 1)


class PolicyFunction:
    def __init__(self, gamma=0.99, lam=0.95):
        self.gamma = gamma
        self.lam = lam

    def monte_carlo_discounted(self, rewards, dones):
        running_add = 0
        returns = []

        for step in reversed(range(len(rewards))):
            running_add = rewards[step] + (1.0 - dones[step]) * self.gamma * running_add
            returns.insert(0, running_add)

        return tf.stack(returns)

    def temporal_difference(self, reward, next_value, done):
        q_values = reward + (1 - done) * self.gamma * next_value
        return q_values

    def generalized_advantage_estimation(self, values, rewards, next_values, dones):
        gae = 0
        adv = []

        delta = rewards + (1.0 - dones) * self.gamma * next_values - values
        for step in reversed(range(len(rewards))):
            gae = delta[step] + (1.0 - dones[step]) * self.gamma * self.lam * gae
            adv.insert(0, gae)

        return tf.stack(adv)


# https://github.com/wisnunugroho21/reinforcement_learning_ppo_rnd
class PPOAgent(Agent):
    def __init__(self, obs_dim, act_dim, policy_kl_range, policy_params, value_clip, entropy_coef, vf_loss_coef,
                 minibatch, PPO_epochs, gamma, lam, learning_rate, n_update, layer_names=None,
                 nb_observations=50000, sparsity=0.5, epsilon=1):
        super().__init__(obs_dim, act_dim)
        self.policy_kl_range = policy_kl_range
        self.policy_params = policy_params
        self.value_clip = value_clip
        self.entropy_coef = entropy_coef
        self.vf_loss_coef = vf_loss_coef
        self.minibatch = minibatch
        self.PPO_epochs = PPO_epochs

        self.actor = Actor_Model(self.obs_dim, self.act_dim)
        self.actor_old = Actor_Model(self.obs_dim, self.act_dim)

        self.critic = Critic_Model(self.obs_dim, self.act_dim)
        self.critic_old = Critic_Model(self.obs_dim, self.act_dim)

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.memory = Memory()
        self.policy_function = PolicyFunction(gamma, lam)
        self.distributions = Distributions()

        # Set hyperparameters
        self.gamma = gamma  # discount factor
        self.lam = lam
        self.n_update = n_update  # number of epochs per episode
        # pruning attributes
        if layer_names is None:
            layer_names = ['Policy_L0', 'Policy_L1']
        self.pruned_actor = None
        self._region_locator = DNNRegionsLocator(self.actor, layer_names=layer_names, nb_observations=nb_observations,
                                                 sparsity=sparsity, epsilon=epsilon)

    def save_eps(self, state, action, reward, done, next_state):
        self.memory.save_eps(state, action, reward, done, next_state)

    # Loss for PPO
    def get_loss(self, action_probs, values, old_action_probs, old_values, next_values, actions, rewards, dones):
        # Don't use old value in backpropagation
        Old_values = tf.stop_gradient(old_values)

        # Getting general advantages estimator
        Advantages = self.policy_function.generalized_advantage_estimation(values, rewards, next_values, dones)
        Returns = tf.stop_gradient(Advantages + values)
        Advantages = tf.stop_gradient(
            (Advantages - tf.math.reduce_mean(Advantages)) / (tf.math.reduce_std(Advantages) + 1e-6))

        # Finding the ratio (pi_theta / pi_theta__old):
        logprobs = self.distributions.logprob(action_probs, actions)
        Old_logprobs = tf.stop_gradient(self.distributions.logprob(old_action_probs, actions))
        ratios = tf.math.exp(logprobs - Old_logprobs)  # ratios = old_logprobs / logprobs

        # Finding KL Divergence
        Kl = self.distributions.kl_divergence(old_action_probs, action_probs)

        # Combining TR-PPO with Rollback (Truly PPO)
        pg_loss = tf.where(
            tf.logical_and(Kl >= self.policy_kl_range, ratios > 1),
            ratios * Advantages - self.policy_params * Kl,
            ratios * Advantages
        )
        pg_loss = tf.math.reduce_mean(pg_loss)

        # Getting entropy from the action probability
        dist_entropy = tf.math.reduce_mean(self.distributions.entropy(action_probs))

        # Getting critic loss by using Clipped critic value
        vpredclipped = old_values + tf.clip_by_value(values - Old_values, -self.value_clip,
                                                     self.value_clip)  # Minimize the difference between old value and new value
        vf_losses1 = tf.math.square(Returns - values) * 0.5  # Mean Squared Error
        vf_losses2 = tf.math.square(Returns - vpredclipped) * 0.5  # Mean Squared Error
        critic_loss = tf.math.reduce_mean(tf.math.maximum(vf_losses1, vf_losses2))

        # We need to maximaze Policy Loss to make agent always find Better Rewards
        # and minimize Critic Loss
        loss = (critic_loss * self.vf_loss_coef) - (dist_entropy * self.entropy_coef) - pg_loss
        return loss

    @tf.function
    def action(self, state, pruning=False):
        state = tf.expand_dims(tf.cast(state, dtype=tf.float32), 0)
        if pruning:
            action_probs = self.pruned_actor(state)
        else:
            action_probs = self.actor(state)

        # We don't need sample the action in Test Mode
        # only sampling the action in Training Mode in order to exploring the actions
        if self._training:
            # Sample the action
            action = self.distributions.sample(action_probs)
        else:
            action = tf.math.argmax(action_probs, 1)

        return action

    # Get loss and Do backpropagation
    @tf.function
    def update(self, data, pruning=False):
        states, actions, rewards, dones, next_states = data

        # if not pruning:
        #     # collect observation to determine major/minor regions (for 'forget mechanism')
        #     self._region_locator.add_observations(states)

        with tf.GradientTape() as tape:
            if pruning:
                action_probs, values = self.pruned_actor(states), self.critic(states)
            else:
                action_probs, values = self.actor(states), self.critic(states)

            old_action_probs, old_values = self.actor_old(states), self.critic_old(states)
            next_values = self.critic(next_states)

            loss = self.get_loss(action_probs, values, old_action_probs, old_values, next_values, actions, rewards,
                                 dones)

        if pruning:
            gradients = tape.gradient(loss, self.pruned_actor.trainable_variables + self.critic.trainable_variables)
            self.optimizer.apply_gradients(
                zip(gradients, self.pruned_actor.trainable_variables + self.critic.trainable_variables))
        else:
            gradients = tape.gradient(loss, self.actor.trainable_variables + self.critic.trainable_variables)
            self.optimizer.apply_gradients(
                zip(gradients, self.actor.trainable_variables + self.critic.trainable_variables))

    def save(self, dir_path):
        actor_path = os.path.join(dir_path, 'actor_ppo')
        critic_path = os.path.join(dir_path, 'critic_ppo')
        actor_old_path = os.path.join(dir_path, 'actor_old_ppo')
        critic_old_path = os.path.join(dir_path, 'critic_old_ppo')
        pruned_actor_file_path = os.path.join(dir_path, 'pruned_actor_ppo')
        self.actor.save_weights(actor_path, save_format='tf')
        self.critic.save_weights(critic_path, save_format='tf')
        self.actor_old.save_weights(actor_old_path, save_format='tf')
        self.critic_old.save_weights(critic_old_path, save_format='tf')
        self._region_locator.save_observations(dir_path)
        if self.pruned_actor is not None:
            self.pruned_actor.save_weights(pruned_actor_file_path, save_format='tf')

    def save_pruned_policy(self, dir_path):
        pruned_actor_file_path = os.path.join(dir_path, 'pruned_actor_ppo')
        if self.pruned_actor is not None:
            self.pruned_actor.save_weights(pruned_actor_file_path, save_format='tf')

    def load(self, dir_path, load_observation=True, load_pruned=False):
        actor_path = os.path.join(dir_path, 'actor_ppo')
        critic_path = os.path.join(dir_path, 'critic_ppo')
        actor_old_path = os.path.join(dir_path, 'actor_old_ppo')
        critic_old_path = os.path.join(dir_path, 'critic_old_ppo')
        self.actor.load_weights(actor_path)
        self.critic.load_weights(critic_path)
        self.actor_old.load_weights(actor_old_path)
        self.critic_old.load_weights(critic_old_path)

        if load_pruned:
            self.pruned_actor = Actor_Model(self.obs_dim, self.act_dim)
            self.pruned_actor.load_weights(actor_path)

        if load_observation:
            self._region_locator.load_observations(dir_path)
        # self._region_locator.set_model(self.actor)
        self._region_locator.set_layer_names([self.actor.layers[0].name, self.actor.layers[1].name])

    def load_pruned_policy(self, dir_path):
        pruned_actor_file_path = os.path.join(dir_path, 'pruned_actor_ppo')
        self.pruned_actor.load_weights(pruned_actor_file_path)

    def setup_pruning(self):

        # self.pruned_actor = Actor_Model(self.obs_dim, self.act_dim)
        # self.pruned_actor.set_weights(self.actor.get_weights())

        input1 = tf.keras.Input(shape=(self.obs_dim), dtype=tf.float32)

        self.pruned_actor(input1)

        # Non-boilerplate.
        self.pruned_actor.optimizer = self.optimizer

        # we need to perform pruning here
        masks = self._region_locator.generate_regions_masks()
        model_weights = self.pruned_actor.get_weights()
        self.pruned_actor = self._region_locator.apply_mask(masks, self.pruned_actor)
        new_model_weights = self.pruned_actor.get_weights()
        print()
