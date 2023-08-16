import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Model


# helper function
def calculate_ats(model, data, layer_names, batch_size=1024):
    out = model(data)
    output_layers = [model.get_layer(layer_name).output for layer_name in layer_names]
    output_layers.append(model.output)
    temp_model = Model(inputs=model.input, outputs=output_layers)

    layers_output = temp_model.predict(data, batch_size=batch_size, verbose=0)
    # Remove the (output layer) dnn outputs from the list and store them as separate result
    layers_output.pop()
    layers_output = [layer_output.reshape((len(data), -1)) for layer_output in layers_output]

    layer_output_dict = {}
    for i in range(len(layer_names)):
        layer_output_dict[layer_names[i]] = layers_output[i]

    return layer_output_dict


class DNNRegionsLocator:
    def __init__(self, model, layer_names, nb_observations, sparsity=0.5, epsilon=0.1):
        self.model = model
        self.observations = []
        self.pattern_set = set()
        self.sparsity = sparsity
        self.layer_names = layer_names
        self.nb_observations = nb_observations
        self.epsilon = epsilon

    def set_model(self, model):
        self.model = model

    def set_layer_names(self, layer_names):
        self.layer_names = layer_names

    def add_observations(self, observations):
        self.observations.append(observations)

    def generate_regions_mask_at_layer(self, ats, layer_weights):

        k = np.round(ats.shape[1] * (1 - self.sparsity)).astype('int32')
        bottom_neurons = np.argpartition(ats, k, axis=1)[:, :k]
        bottom_neurons = bottom_neurons.flatten()
        sorted_bottom_neurons, bottom_neuron_appearances = np.unique(bottom_neurons, return_counts=True)
        frequent_bottom_neuron_indices = np.argsort(-bottom_neuron_appearances)
        frequent_bottom_neuron = sorted_bottom_neurons[frequent_bottom_neuron_indices]
        frequent_k_bottom_neuron = frequent_bottom_neuron[:k]

        mask = np.ones(shape=layer_weights.shape)
        mask[:, frequent_k_bottom_neuron] = 0

        return mask

    def generate_regions_mask_at_layer_2(self, ats, layer_weights):

        k = np.round(ats.shape[1] * (1 - self.sparsity)).astype('int32')
        top_neurons = np.argpartition(ats, k, axis=1)[:, k:]
        top_neurons = top_neurons.flatten()
        sorted_top_neurons, top_neuron_appearances = np.unique(top_neurons, return_counts=True)
        frequent_top_neuron_indices = np.argsort(top_neuron_appearances)
        frequent_top_neuron = sorted_top_neurons[frequent_top_neuron_indices]
        frequent_k_top_neuron = frequent_top_neuron[:k]

        mask = np.ones(shape=layer_weights.shape)
        mask[:, frequent_k_top_neuron] = 0

        return mask

    def generate_regions_masks(self, batch_size=1024):
        masks = []
        # all_observations = np.vstack(self.observations)
        if len(self.observations) > self.nb_observations:
            chosen_observation_indices = np.random.choice(len(self.observations), self.nb_observations, replace=False)
        else:
            chosen_observation_indices = np.random.choice(len(self.observations), len(self.observations), replace=False)
        observations = self.observations[chosen_observation_indices]

        layers_output_dict = calculate_ats(self.model, observations, self.layer_names, batch_size)
        # layers_output_dict = calculate_ats_poo_2(self.model, observations, self.layer_names, batch_size)
        a = self.model.layers
        for i, layer in enumerate(self.model.layers):
            if layer.name not in self.layer_names:
                continue
            mask = self.generate_regions_mask_at_layer(layers_output_dict[layer.name], layer.weights[0].numpy())
            masks.append(mask)
        return masks

    def apply_mask(self, masks, pruned_model):
        model_weights = self.model.get_weights()
        count = 0
        layers = self.model.layers[1:] if 'input' in self.model.layers[0].name.lower() else self.model.layers
        for i, layer in enumerate(layers):
            if layer.name not in self.layer_names:
                continue
            weight_mask = masks[count]
            weight_mask_inverted = np.logical_not(weight_mask)
            # bias_mask = masks[count][0]
            # bias_mask_inverted = np.logical_not(bias_mask)

            layer_weights = layer.weights
            weight_initializer = layer.kernel_initializer
            # bias_initializer = layer.bias_initializer

            masked_weight = tf.dtypes.cast(
                tf.math.multiply(layer_weights[0], weight_mask), dtype=layer_weights[0].dtype
            ).numpy()

            # masked_bias = tf.dtypes.cast(
            #     tf.math.multiply(layer_weights[1], bias_mask), dtype=layer_weights[1].dtype
            # ).numpy()

            initialized_weights = np.array(weight_initializer(shape=layer_weights[0].shape))
            initialized_weights = initialized_weights * self.epsilon
            # initialized_bias = np.array(bias_initializer(shape=layer_weights[1].shape))

            masked_initialized_weights = tf.dtypes.cast(
                tf.math.multiply(initialized_weights, weight_mask_inverted), dtype=initialized_weights.dtype
            ).numpy()
            # masked_initialized_bias = tf.dtypes.cast(
            #     tf.math.multiply(initialized_bias, bias_mask_inverted), dtype=initialized_bias.dtype
            # ).numpy()

            new_weights = masked_weight + masked_initialized_weights
            # new_bias = masked_bias + masked_initialized_bias

            model_weights[i * 2] = new_weights
            # model_weights[(i * 2) + 1] = new_bias

            count += 1
        pruned_model.set_weights(model_weights)
        print()
        return pruned_model

    def save_observations(self, dir_path):
        file_path = os.path.join(dir_path, 'all_observations.npy')
        self.observations = np.vstack(self.observations)
        np.save(file_path, self.observations)

    def load_observations(self, dir_path):
        file_path = os.path.join(dir_path, 'all_observations.npy')
        self.observations = np.load(file_path)


