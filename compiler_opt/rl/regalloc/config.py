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
"""Register allocation training config."""

from numpy import dtype, maximum, minimum
import gin
import tensorflow as tf
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import time_step
from compiler_opt.rl import feature_ops

import tensorflow_transform as tft

from absl import logging

def get_num_registers():
  return 33

def get_opcode_count():
  return 17716

def get_num_instructions():
  return 300

class process_instruction_features(tf.keras.Model):

  def __init__(self, opcode_count, register_count, instruction_count, mbb_quantiles, embedding_dimensions=16):
    super().__init__()
    self.opcode_count = opcode_count
    self.register_count = register_count
    self.instruction_count = instruction_count
    self.embedding_dimensions = embedding_dimensions
    self.embedding_layer = tf.keras.layers.Embedding(self.opcode_count,
                                                     self.embedding_dimensions,
                                                     input_length=self.instruction_count)
    self.mbb_quantiles = mbb_quantiles
    self.mbb_embedding_layer = tf.keras.layers.Embedding(1000,
                                                         4,
                                                         input_length=self.instruction_count)

  def get_config(self):
    return {
      "opcode_count": self.opcode_count,
      "register_count": self.register_count,
      "instruction_count": self.instruction_count,
      "embedding_dimensions": self.embedding_dimensions,
      'mbb_quantiles': self.mbb_quantiles
    }

  @classmethod
  def from_config(cls, config):
    return cls(**config)

  def call(self, inputs):
    instruction_opcodes = inputs[0]
    instruction_opcodes = tf.reshape(instruction_opcodes, [-1, self.instruction_count])
    binary_mapping_matrix = inputs[1]
    binary_mapping_matrix_casted = tf.cast(binary_mapping_matrix, tf.float32)
    instruction_embeddings = self.embedding_layer(instruction_opcodes)

    matrix_product = tf.linalg.matmul(binary_mapping_matrix_casted,
                                      instruction_embeddings)

    mbb_frequencies = tf.reshape(inputs[2], [-1, self.instruction_count])
    mbb_quantiles = tft.apply_buckets(mbb_frequencies, [self.mbb_quantiles])
    embedded_mbb_frequencies = self.mbb_embedding_layer(mbb_quantiles)
    mbb_matrix_product = tf.linalg.matmul(binary_mapping_matrix_casted, embedded_mbb_frequencies)

    concatenated_products = tf.concat([matrix_product, mbb_matrix_product],
                                      axis=2)

    return concatenated_products

# pylint: disable=g-complex-comprehension
@gin.configurable()
def get_regalloc_signature_spec():
  """Returns (time_step_spec, action_spec) for LLVM register allocation."""
  # LINT.IfChange
  num_registers = get_num_registers()
  num_instructions = get_num_instructions()
  num_opcodes = get_opcode_count()

  observation_spec = dict(
      (key, tf.TensorSpec(dtype=tf.int64, shape=(num_registers), name=key))
      for key in ['mask'])
  observation_spec['instructions'] = tensor_spec.BoundedTensorSpec(
      dtype=tf.int64, shape=(num_instructions), 
      name='instructions', minimum=0, maximum=num_opcodes)
  observation_spec['instructions_mapping'] = tensor_spec.BoundedTensorSpec(
    dtype=tf.int64, shape=(num_registers, num_instructions),
    name='instructions_mapping', minimum=0, maximum=1)
  observation_spec['mbb_frequencies'] = tensor_spec.BoundedTensorSpec(
    dtype=tf.float32, shape=(num_instructions),
    name='mbb_frequencies', minimum=0, maximum=1)

  reward_spec = tf.TensorSpec(dtype=tf.float32, shape=(), name='reward')
  time_step_spec = time_step.time_step_spec(observation_spec, reward_spec)

  action_spec = tensor_spec.BoundedTensorSpec(
      dtype=tf.int64,
      shape=(),
      name='index_to_evict',
      minimum=0,
      maximum=num_registers - 1)
  
  multi_input_preprocessing_layers = [
    ('instructions', 'instructions_mapping', 'mbb_frequencies')
  ]

  return time_step_spec, action_spec, multi_input_preprocessing_layers
  # LINT.ThenChange(.../rl/regalloc/sparse_bucket_config.pbtxt)


@gin.configurable
def get_observation_processing_layer_creator(quantile_file_dir=None,
                                             with_sqrt=True,
                                             with_z_score_normalization=True,
                                             eps=1e-8):
  """Wrapper for observation_processing_layer."""
  quantile_map = feature_ops.build_quantile_map(quantile_file_dir)

  def observation_processing_layer(obs_spec):
    """Creates the layer to process observation given obs_spec."""
    if obs_spec == ('instructions', 'instructions_mapping', 'mbb_frequencies'):
      return process_instruction_features(get_opcode_count(),
                                          get_num_instructions(),
                                          get_num_instructions(),
                                          quantile_map['mbb_frequencies'])

    if obs_spec in ('mask'):
      return tf.keras.layers.Lambda(feature_ops.discard_fn)

    # Make sure all features have a preprocessing function.
    raise KeyError('Missing preprocessing function for some feature.')

  return observation_processing_layer

def get_nonnormalized_features():
  return ['mask', 'nr_urgent',
          'is_hint', 'is_local',
          'is_free', 'max_stage',
          'min_stage', 'reward',
          'instructions', 'instructions_mapping']

def flags_to_add():
  return ('-mllvm', '-regalloc-enable-development-features')