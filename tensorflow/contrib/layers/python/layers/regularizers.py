# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Regularizers for use with layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numbers

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import standard_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.ops.gen_array_ops import reshape

__all__ = ['l1_regularizer',
           'l2_regularizer',
           'l1_l2_regularizer',
           'l2_path_regularizer',
           'sum_regularizer',
           'apply_regularization']

# Decorator to distinguish regularizers that need all the weights at once
def needs_all_weights(regularizer):
  regularizer.depends_on_all_weights = True
  return regularizer

# https://arxiv.org/pdf/1506.02617.pdf
# According to the authors, this regularizer may be useful for ReLU units,
# due to it's invariant nature to rescaling.
# NOTE: 
#   The current implementation works for fully connected networks.
# TODO: 
#   Extend the implementation to partially connected nets by
#   multiplying the weights only along the paths that connect 
#   the input to the output.
def l2_path_regularizer(scale, scope=None):
  """Returns a function that can be used to apply L2 path regularization to weights.

  Args:
    scale: A scalar multiplier `Tensor`. 0.0 disables the regularizer.
    scope: An optional scope name.

  Returns:
    A function with signature `l2_path(weights)` that apply L2 path regularization.

  Raises:
    ValueError: If scale is negative or if scale is not a float.
  """
  if isinstance(scale, numbers.Integral):
    raise ValueError('scale cannot be an integer: %s' % scale)
  if isinstance(scale, numbers.Real):
    if scale < 0.:
      raise ValueError('Setting a scale less than 0 on a regularizer: %g' %
                       scale)
    if scale == 0.:
      logging.info('Scale of 0 disables regularizer.')
      return lambda _: None

  @needs_all_weights
  def l2_path(weights_list, name=None):
    """Applies L2 path regularization to weights.
    
    Args:
      weights_list: A list of weight tensors accross the layers of
                    the fully connected neural net.
    """

    with ops.name_scope(scope, 'l2_path_regularizer', weights_list) as name:
      # We require the input to the regularizer to be a list of weights
      if type(weights_list) is not list or not weights_list:
        raise TypeError('A list of the weights is required.')

      # Cast the data types of `my_scale`, `initializer`, 'rescaler' to that of the weights
      dtype = weights_list[0].dtype.base_dtype
      my_scale = ops.convert_to_tensor(scale,
                                       dtype=dtype,
                                       name='scale')

      initializer = constant_op.constant(1.,
                                         dtype=dtype,
                                         name='initializer')

      # Since L2LossOp multiplies the L2 regularization of each weight tensor by 0.5,
      # our final result will be off by a factor of 0.5 ^ (len(weights_list)-1).
      rescaler = constant_op.constant(pow(2, len(weights_list)-1),
                                      dtype=dtype,
                                      name='rescaler')

      # The sum of the scaled weights along the input->output paths on the graph
      scaled_reduced_path_products = standard_ops.foldl(
                                         lambda x, y: standard_ops.multiply(x, nn.l2_loss(y)), 
                                         weights_list, 
                                         initializer=initializer)

      # We apply the `rescaler` to obtain a desired scale factor of 0.5
      reduced_path_products = standard_ops.multiply(
          rescaler,
          scaled_reduced_path_products
        )

      # Apply the desired amount of regularization
      return standard_ops.multiply(
        my_scale,
        reduced_path_products, name=name)

  return l2_path

def l1_regularizer(scale, scope=None):
  """Returns a function that can be used to apply L1 regularization to weights.

  L1 regularization encourages sparsity.

  Args:
    scale: A scalar multiplier `Tensor`. 0.0 disables the regularizer.
    scope: An optional scope name.

  Returns:
    A function with signature `l1(weights)` that apply L1 regularization.

  Raises:
    ValueError: If scale is negative or if scale is not a float.
  """
  if isinstance(scale, numbers.Integral):
    raise ValueError('scale cannot be an integer: %s' % scale)
  if isinstance(scale, numbers.Real):
    if scale < 0.:
      raise ValueError('Setting a scale less than 0 on a regularizer: %g' %
                       scale)
    if scale == 0.:
      logging.info('Scale of 0 disables regularizer.')
      return lambda _: None

  def l1(weights, name=None):
    """Applies L1 regularization to weights."""
    with ops.name_scope(scope, 'l1_regularizer', [weights]) as name:
      my_scale = ops.convert_to_tensor(scale,
                                       dtype=weights.dtype.base_dtype,
                                       name='scale')
      return standard_ops.multiply(
          my_scale,
          standard_ops.reduce_sum(standard_ops.abs(weights)),
          name=name)

  return l1


def l2_regularizer(scale, scope=None):
  """Returns a function that can be used to apply L2 regularization to weights.

  Small values of L2 can help prevent overfitting the training data.

  Args:
    scale: A scalar multiplier `Tensor`. 0.0 disables the regularizer.
    scope: An optional scope name.

  Returns:
    A function with signature `l2(weights)` that applies L2 regularization.

  Raises:
    ValueError: If scale is negative or if scale is not a float.
  """
  if isinstance(scale, numbers.Integral):
    raise ValueError('scale cannot be an integer: %s' % (scale,))
  if isinstance(scale, numbers.Real):
    if scale < 0.:
      raise ValueError('Setting a scale less than 0 on a regularizer: %g.' %
                       scale)
    if scale == 0.:
      logging.info('Scale of 0 disables regularizer.')
      return lambda _: None

  def l2(weights):
    """Applies l2 regularization to weights."""
    with ops.name_scope(scope, 'l2_regularizer', [weights]) as name:
      my_scale = ops.convert_to_tensor(scale,
                                       dtype=weights.dtype.base_dtype,
                                       name='scale')
      return standard_ops.multiply(my_scale, nn.l2_loss(weights), name=name)

  return l2


def l1_l2_regularizer(scale_l1=1.0, scale_l2=1.0, scope=None):
  """Returns a function that can be used to apply L1 L2 regularizations.

  Args:
    scale_l1: A scalar multiplier `Tensor` for L1 regularization.
    scale_l2: A scalar multiplier `Tensor` for L2 regularization.
    scope: An optional scope name.

  Returns:
    A function with signature `l1_l2(weights)` that applies a weighted sum of
    L1 L2  regularization.

  Raises:
    ValueError: If scale is negative or if scale is not a float.
  """
  scope = scope or 'l1_l2_regularizer'
  return sum_regularizer([l1_regularizer(scale_l1),
                          l2_regularizer(scale_l2)],
                         scope=scope)


def sum_regularizer(regularizer_list, scope=None):
  """Returns a function that applies the sum of multiple regularizers.

  Args:
    regularizer_list: A list of regularizers to apply.
    scope: An optional scope name

  Returns:
    A function with signature `sum_reg(weights)` that applies the
    sum of all the input regularizers.
  """
  regularizer_list = [reg for reg in regularizer_list if reg is not None]
  if not regularizer_list:
    return None

  def sum_reg(weights):
    """Applies the sum of all the input regularizers."""
    with ops.name_scope(scope, 'sum_regularizer', [weights]) as name:
      regularizer_tensors = [reg(weights) for reg in regularizer_list]
      return math_ops.add_n(regularizer_tensors, name=name)

  return sum_reg


def apply_regularization(regularizer, weights_list=None):
  """Returns the summed penalty by applying `regularizer` to the `weights_list`.

  Adding a regularization penalty over the layer weights and embedding weights
  can help prevent overfitting the training data. Regularization over layer
  biases is less common/useful, but assuming proper data preprocessing/mean
  subtraction, it usually shouldn't hurt much either.

  Args:
    regularizer: A function that takes a single `Tensor` argument and returns
      a scalar `Tensor` output.
    weights_list: List of weights `Tensors` or `Variables` to apply
      `regularizer` over. Defaults to the `GraphKeys.WEIGHTS` collection if
      `None`.

  Returns:
    A scalar representing the overall regularization penalty.

  Raises:
    ValueError: If `regularizer` does not return a scalar output, or if we find
        no weights.
  """
  if not weights_list:
    weights_list = ops.get_collection(ops.GraphKeys.WEIGHTS)
  if not weights_list:
    raise ValueError('No weights to regularize.')
  with ops.name_scope('get_regularization_penalty',
                      values=weights_list) as scope:
    # Some regularizations are not (simple) linear combinations of the weights
    # and require all of the weights at once.
    if hasattr(regularizer, 'depends_on_all_weights'):
      summed_penalty = regularizer(weights_list, name=scope)
      if summed_penalty.get_shape().ndims != 0:
        raise ValueError('regularizer must return a scalar Tensor instead of a '
                         'Tensor with rank %d.' % p.get_shape().ndims)
    else:
      penalties = [regularizer(w) for w in weights_list]
      penalties = [
          p if p is not None else constant_op.constant(0.0) for p in penalties
      ]
      for p in penalties:
        if p.get_shape().ndims != 0:
          raise ValueError('regularizer must return a scalar Tensor instead of a '
                           'Tensor with rank %d.' % p.get_shape().ndims)

      summed_penalty = math_ops.add_n(penalties, name=scope)

    ops.add_to_collection(ops.GraphKeys.REGULARIZATION_LOSSES, summed_penalty)
    return summed_penalty
