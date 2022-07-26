# coding=utf-8
# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Actor network for Register Allocation."""

from typing import Optional, Sequence, Callable, Text, Any

import gin
import tensorflow as tf
from tf_agents.networks import categorical_projection_network
from tf_agents.networks import encoding_network
from tf_agents.networks import network
from tf_agents.typing import types
from tf_agents.utils import nest_utils

from absl import logging

from tensorflow.python.util import nest
from tf_agents.keras_layers import permanent_variable_rate_dropout
from tf_agents.networks import utils

CONV_TYPE_2D = '2d'
CONV_TYPE_1D = '1d'


def _copy_layer(layer):
  """Create a copy of a Keras layer with identical parameters.
  The new layer will not share weights with the old one.
  Args:
    layer: An instance of `tf.keras.layers.Layer`.
  Returns:
    A new keras layer.
  Raises:
    TypeError: If `layer` is not a keras layer.
    ValueError: If `layer` cannot be correctly cloned.
  """
  if not isinstance(layer, tf.keras.layers.Layer):
    raise TypeError('layer is not a keras layer: %s' % str(layer))

  # pylint:disable=unidiomatic-typecheck
  if type(layer) == tf.compat.v1.keras.layers.DenseFeatures:
    raise ValueError('DenseFeatures V1 is not supported. '
                     'Use tf.compat.v2.keras.layers.DenseFeatures instead.')
  if layer.built:
    logging.warning(
        'Beware: Copying a layer that has already been built: \'%s\'.  '
        'This can lead to subtle bugs because the original layer\'s weights '
        'will not be used in the copy.', layer.name)
  # Get a fresh copy so we don't modify an incoming layer in place.  Weights
  # will not be shared.
  return type(layer).from_config(layer.get_config())


class RegAllocEncodingNetwork(encoding_network.EncodingNetwork):

  def __init__(self,
               input_tensor_spec,
               preprocessing_layers=None,
               preprocessing_combiner=None,
               conv_layer_params=None,
               fc_layer_params=None,
               dropout_layer_params=None,
               activation_fn=tf.keras.activations.relu,
               weight_decay_params=None,
               kernel_initializer=None,
               batch_squash=True,
               dtype=tf.float32,
               name='EncodingNetwork',
               conv_type=CONV_TYPE_2D):
    if preprocessing_layers is None:
      flat_preprocessing_layers = None
      preprocessing_nest = None
    else:
      # tf.nest.flatten doesn't support tuples as dict keys
      if isinstance(preprocessing_layers, dict):
        flat_preprocessing_layers = []
        preprocessing_nest = {}
        for layer in preprocessing_layers:
          preprocessing_nest[layer] = len(flat_preprocessing_layers)
          flat_preprocessing_layers.append(
              _copy_layer(preprocessing_layers[layer]))
      else:
        flat_preprocessing_layers = [
            _copy_layer(layer)
            for layer in tf.nest.flatten(preprocessing_layers)
        ]
        preprocessing_nest = tf.nest.map_structure(lambda l: None,
                                                   preprocessing_layers)
      # Assert shallow structure is the same. This verifies preprocessing
      # layers can be applied on expected input nests.
      if isinstance(preprocessing_layers, dict):
        layer_inputs = []
        for layer in preprocessing_layers:
          if isinstance(layer, tuple):
            for input_name in layer:
              layer_inputs.append(input_name)
          else:
            layer_inputs.append(layer)
        if len(layer_inputs) != len(input_tensor_spec):
          raise ValueError('the number of inputs to preprocessing layers needs'
                           'to be equal to the number if input tensors')
        for input in layer_inputs:
          if input not in input_tensor_spec:
            raise ValueError('a preprocessing layer requires an input tensor'
                             f'{input}, but it is not present')
      else:
        input_nest = input_tensor_spec
        # Given the flatten on preprocessing_layers above we need to make sure
        # input_tensor_spec is a sequence for the shallow_structure check below
        # to work.
        if not nest.is_sequence(input_tensor_spec):
          input_nest = [input_tensor_spec]
        nest.assert_shallow_structure(preprocessing_layers, input_nest)

    if (len(tf.nest.flatten(input_tensor_spec)) > 1 and
        preprocessing_combiner is None):
      raise ValueError(
          'preprocessing_combiner layer is required when more than 1 '
          'input_tensor_spec is provided.')

    if preprocessing_combiner is not None:
      preprocessing_combiner = _copy_layer(preprocessing_combiner)

    if not kernel_initializer:
      kernel_initializer = tf.compat.v1.variance_scaling_initializer(
          scale=2.0, mode='fan_in', distribution='truncated_normal')

    layers = []

    if conv_layer_params:
      if conv_type == '2d':
        conv_layer_type = tf.keras.layers.Conv2D
      elif conv_type == '1d':
        conv_layer_type = tf.keras.layers.Conv1D
      else:
        raise ValueError('unsupported conv type of %s. Use 1d or 2d' %
                         (conv_type))

      for config in conv_layer_params:
        if len(config) == 4:
          (filters, kernel_size, strides, dilation_rate) = config
        elif len(config) == 3:
          (filters, kernel_size, strides) = config
          dilation_rate = (1, 1) if conv_type == '2d' else (1,)
        else:
          raise ValueError(
              'only 3 or 4 elements permitted in conv_layer_params tuples')
        layers.append(
            conv_layer_type(
                filters=filters,
                kernel_size=kernel_size,
                strides=strides,
                dilation_rate=dilation_rate,
                activation=activation_fn,
                kernel_initializer=kernel_initializer,
                dtype=dtype))

    layers.append(tf.keras.layers.Flatten())

    if fc_layer_params:
      if dropout_layer_params is None:
        dropout_layer_params = [None] * len(fc_layer_params)
      else:
        if len(dropout_layer_params) != len(fc_layer_params):
          raise ValueError('Dropout and fully connected layer parameter lists'
                           'have different lengths (%d vs. %d.)' %
                           (len(dropout_layer_params), len(fc_layer_params)))
      if weight_decay_params is None:
        weight_decay_params = [None] * len(fc_layer_params)
      else:
        if len(weight_decay_params) != len(fc_layer_params):
          raise ValueError('Weight decay and fully connected layer parameter '
                           'lists have different lengths (%d vs. %d.)' %
                           (len(weight_decay_params), len(fc_layer_params)))

      for num_units, dropout_params, weight_decay in zip(
          fc_layer_params, dropout_layer_params, weight_decay_params):
        kernal_regularizer = None
        if weight_decay is not None:
          kernal_regularizer = tf.keras.regularizers.l2(weight_decay)
        layers.append(
            tf.keras.layers.Dense(
                num_units,
                activation=activation_fn,
                kernel_initializer=kernel_initializer,
                kernel_regularizer=kernal_regularizer,
                dtype=dtype))
        if not isinstance(dropout_params, dict):
          dropout_params = {'rate': dropout_params} if dropout_params else None

        if dropout_params is not None:
          layers.append(
              permanent_variable_rate_dropout.PermanentVariableRateDropout(
                  **dropout_params))

    super(encoding_network.EncodingNetwork, self).__init__(
        input_tensor_spec=input_tensor_spec, state_spec=(), name=name)

    # Set preprocessing_nest directly so that keras doesn't do any
    # processing to it and error out when tf.nest.flatten is called
    self.__dict__['_preprocessing_nest'] = preprocessing_nest
    self._flat_preprocessing_layers = flat_preprocessing_layers
    self._preprocessing_combiner = preprocessing_combiner
    self._postprocessing_layers = layers
    self._batch_squash = batch_squash
    self.built = True  # Allow access to self.variables
    self._postprocessing_layers = self._postprocessing_layers[1:]

  def call(self, observation, step_type=None, network_state=(), training=False):
    del step_type  # unused.

    if self._batch_squash:
      outer_rank = nest_utils.get_outer_rank(observation,
                                             self.input_tensor_spec)
      batch_squash = utils.BatchSquash(outer_rank)
      observation = tf.nest.map_structure(batch_squash.flatten, observation)

    if self._flat_preprocessing_layers is None:
      processed = observation
    else:
      processed = []
      if isinstance(self._preprocessing_nest, dict):
        for layer_name in self._preprocessing_nest:
          preprocessing_layer = self._flat_preprocessing_layers[
              self._preprocessing_nest[layer_name]]
          if isinstance(layer_name, tuple):
            print("called layer on tuple")
            needed_inputs = []
            for input_name in layer_name:
              needed_inputs.append(observation[input_name])
            processed.append(
                preprocessing_layer(needed_inputs, training=training))
          else:
            processed.append(
                preprocessing_layer(observation[layer_name], training=training))
      else:
        for obs, layer in zip(
            nest.flatten_up_to(self._preprocessing_nest, observation),
            self._flat_preprocessing_layers):
          processed.append(layer(obs, training=training))
      if len(processed) == 1 and self._preprocessing_combiner is None:
        # If only one observation is passed and the preprocessing_combiner
        # is unspecified, use the preprocessed version of this observation.
        processed = processed[0]

    states = processed

    if self._preprocessing_combiner is not None:
      states = self._preprocessing_combiner(states)

    for layer in self._postprocessing_layers:
      states = layer(states, training=training)

    if self._batch_squash:
      states = tf.nest.map_structure(batch_squash.unflatten, states)

    return states, network_state


class RegAllocProbProjectionNetwork(
    categorical_projection_network.CategoricalProjectionNetwork):

  def __init__(self, **kwargs):
    super().__init__(**kwargs)
    # shape after projection_layer: B x T x 33 x 1; then gets re-shaped to
    # B x T x 33.
    self._projection_layer = tf.keras.layers.Dense(
        1,
        kernel_initializer=tf.compat.v1.keras.initializers.VarianceScaling(
            scale=kwargs['logits_init_output_factor']),
        bias_initializer=tf.keras.initializers.Zeros(),
        name='logits')


@gin.configurable
class RegAllocRNDEncodingNetwork(RegAllocEncodingNetwork):

  def __init__(self, **kwargs):
    pooling_layer = tf.keras.layers.GlobalMaxPool1D(data_format='channels_last')
    super().__init__(**kwargs)
    # add a pooling layer at the end to to convert B x T x 33 x dim to
    # B x T x dim.
    self._postprocessing_layers.append(pooling_layer)


@gin.configurable
class RegAllocNetwork(network.DistributionNetwork):
  """Creates the actor network for register allocation policy training."""

  def __init__(
      self,
      input_tensor_spec: types.NestedTensorSpec,
      output_tensor_spec: types.NestedTensorSpec,
      preprocessing_layers: Optional[types.NestedLayer] = None,
      preprocessing_combiner: Optional[tf.keras.layers.Layer] = None,
      conv_layer_params: Optional[Sequence[Any]] = None,
      fc_layer_params: Optional[Sequence[int]] = (200, 100),
      dropout_layer_params: Optional[Sequence[float]] = None,
      activation_fn: Callable[[types.Tensor],
                              types.Tensor] = tf.keras.activations.relu,
      kernel_initializer: Optional[tf.keras.initializers.Initializer] = None,
      batch_squash: bool = True,
      dtype: tf.DType = tf.float32,
      name: Text = 'RegAllocNetwork'):
    """Creates an instance of `RegAllocNetwork`.

    Args:
      input_tensor_spec: A nest of `tensor_spec.TensorSpec` representing the
        input.
      output_tensor_spec: A nest of `tensor_spec.BoundedTensorSpec` representing
        the output.
      preprocessing_layers: (Optional.) A nest of `tf.keras.layers.Layer`
        representing preprocessing for the different observations.
        All of these layers must not be already built. For more details see
        the documentation of `networks.EncodingNetwork`.
      preprocessing_combiner: (Optional.) A keras layer that takes a flat list
        of tensors and combines them. Good options include
        `tf.keras.layers.Add` and `tf.keras.layers.Concatenate(axis=-1)`.
        This layer must not be already built. For more details see
        the documentation of `networks.EncodingNetwork`.
      conv_layer_params: Optional list of convolution layers parameters, where
        each item is a length-three tuple indicating (filters, kernel_size,
        stride).
      fc_layer_params: Optional list of fully_connected parameters, where each
        item is the number of units in the layer.
      dropout_layer_params: Optional list of dropout layer parameters, each item
        is the fraction of input units to drop or a dictionary of parameters
        according to the keras.Dropout documentation. The additional parameter
        `permanent`, if set to True, allows to apply dropout at inference for
        approximated Bayesian inference. The dropout layers are interleaved with
        the fully connected layers; there is a dropout layer after each fully
        connected layer, except if the entry in the list is None. This list must
        have the same length of fc_layer_params, or be None.
      activation_fn: Activation function, e.g. tf.nn.relu, slim.leaky_relu, ...
      kernel_initializer: Initializer to use for the kernels of the conv and
        dense layers. If none is provided a default glorot_uniform.
      batch_squash: If True the outer_ranks of the observation are squashed into
        the batch dimension. This allow encoding networks to be used with
        observations with shape [BxTx...].
      dtype: The dtype to use by the convolution and fully connected layers.
      name: A string representing name of the network.

    Raises:
      ValueError: If `input_tensor_spec` contains more than one observation.
    """

    if not kernel_initializer:
      kernel_initializer = tf.compat.v1.keras.initializers.glorot_uniform()

    # input: B x T x obs_spec
    # output: B x T x 33 x dim
    encoder = RegAllocEncodingNetwork(
        input_tensor_spec=input_tensor_spec,
        preprocessing_layers=preprocessing_layers,
        preprocessing_combiner=preprocessing_combiner,
        conv_layer_params=conv_layer_params,
        fc_layer_params=fc_layer_params,
        dropout_layer_params=dropout_layer_params,
        activation_fn=activation_fn,
        kernel_initializer=kernel_initializer,
        batch_squash=batch_squash,
        dtype=dtype)

    projection_network = RegAllocProbProjectionNetwork(
        sample_spec=output_tensor_spec, logits_init_output_factor=0.1)
    output_spec = projection_network.output_spec

    super().__init__(
        input_tensor_spec=input_tensor_spec,
        state_spec=(),
        output_spec=output_spec,
        name=name)

    self._encoder = encoder
    self._projection_network = projection_network
    self._output_tensor_spec = output_tensor_spec

  @property
  def output_tensor_spec(self):
    return self._output_tensor_spec

  def call(self,
           observations: types.NestedTensor,
           step_type: types.NestedTensor,
           network_state=(),
           training: bool = False,
           mask=None):
    _ = mask
    state, network_state = self._encoder(
        observations,
        step_type=step_type,
        network_state=network_state,
        training=training)
    outer_rank = nest_utils.get_outer_rank(observations, self.input_tensor_spec)

    # mask un-evictable registers.
    distribution, _ = self._projection_network(
        state, outer_rank, training=training, mask=observations['mask'])

    return distribution, network_state
