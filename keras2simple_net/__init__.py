#################################################################
# simple_neural_nets
# DNN implementation only relying on a Python base distribution.
# 2024/08/12
# (C) Andreas Gaiser (doraeneko@github)
# Helpers to transform keras nets to simple_neural_nets nets.
# Only the following layer kinds / models and non-default hyperparameters are currently
# supported (activation function is always RELU if not otherwise stated):
# - Dense (size of layer, RELU / linear activation)
# - Conv2D (input channels, number of filters; stride is always 1)
# - BatchNormalization
# - MaxPooling2D (pool size; strides)
# - Rescaling (scale)
# - Flatten
# - Sequential
#################################################################

__all__ = ["transfer_sequential"]

import keras
import numpy as np
import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense,
    Conv2D,
    BatchNormalization,
    MaxPooling2D,
    Rescaling,
    Flatten,
)
from ..simple_net.vector import Vector
from ..simple_net.tensor3d import Tensor3D
from ..simple_net.dense1d_layer import Dense1DLayer
from ..simple_net.conv2d_layer import Conv2DLayer
from ..simple_net.batch_normalization_layer import BatchNormalizationLayer
from ..simple_net.maxpooling2d_layer import MaxPooling2DLayer
from ..simple_net.rescaling_layer import RescalingLayer
from ..simple_net.sequential import Sequential as SimpleNetSequential
from ..simple_net.flatten_layer import FlattenLayer


def transfer_np_vector_to_vector(np_vector, simple_net_vector: Vector):
    for i in range(np_vector.shape[0]):
        simple_net_vector[i] = np_vector[i]


def transfer_batch_size_1_np_tensor_to_tensor3d(np_tensor):
    assert len(np_tensor.shape) == 4 and np_tensor.shape[0] == 1
    (_, x, y, c) = np_tensor.shape
    result = Tensor3D(c, x, y)
    for c_i in range(c):
        for x_i in range(x):
            for y_i in range(y):
                result[c_i, x_i, y_i] = np_tensor[0, x_i, y_i, c_i]
    return result


def extend_2d_np_tensor_to_4d(np_tensor):
    assert len(np_tensor.shape) == 2
    return np_tensor.reshape(1, 1, -1, 1)


def transfer_dense_layer(dense_layer: Dense) -> Dense1DLayer:
    weights, biases = dense_layer.get_weights()
    input_dim, output_dim = weights.shape
    d = Dense1DLayer(
        input_dim,
        output_dim,
        apply_relu=dense_layer.activation == tensorflow.keras.activations.relu,
    )
    i = 0
    for col_vector in weights:
        j = 0
        for row_entry in col_vector:
            d.weights()[i, j] = row_entry
            j += 1
        i += 1
    i = 0
    for bias in biases:
        d.biases()[i] = bias
        i += 1
    return d


def transfer_conv2d_layer(conv2d_layer: Conv2D):
    weights, biases = conv2d_layer.get_weights()
    (filter_height, filter_width, input_channels, num_filters) = weights.shape
    result = Conv2DLayer(input_channels, num_filters, (filter_height, filter_width))
    # transfer biases
    for index, bias in enumerate(biases):
        result.filter(index).set_bias(bias)
    # transfer filter weights
    for filter_index in range(num_filters):
        for channel_index in range(input_channels):
            for i in range(filter_height):
                for j in range(filter_width):
                    result.filter(filter_index).weights(channel_index)[i, j] = weights[
                        i, j, channel_index, filter_index
                    ]
    return result


def transfer_batch_normalization_layer(batch_layer: BatchNormalization):
    (gammas, betas, moving_means, moving_variances) = batch_layer.get_weights()
    result = BatchNormalizationLayer(len(gammas))
    transfer_np_vector_to_vector(gammas, result.gammas())
    transfer_np_vector_to_vector(betas, result.betas())
    transfer_np_vector_to_vector(moving_means, result.running_means())
    transfer_np_vector_to_vector(moving_variances, result.running_variances())
    return result


def transfer_maxpooling2d_layer(pool_layer: MaxPooling2D):
    return MaxPooling2DLayer(pool_size=pool_layer.pool_size, strides=pool_layer.strides)


def transfer_sequential(seq: Sequential) -> SimpleNetSequential:
    """Create a sequential net from the given Sequential in keras format.
    Unsupported layers are simply skipped (which can result in a faulty net).
    No check is done whether a hyperparameter of a
    partially supported net is not yet supported."""
    result = SimpleNetSequential()
    for layer in seq.layers:
        # print("Trying to add %s." % layer)
        if isinstance(layer, BatchNormalization):
            result.push_layer(transfer_batch_normalization_layer(layer))
        elif isinstance(layer, Conv2D):
            result.push_layer(transfer_conv2d_layer(layer))
        elif isinstance(layer, Dense):
            result.push_layer(transfer_dense_layer(layer))
        elif isinstance(layer, Rescaling):
            result.push_layer(RescalingLayer(layer.scale))
        elif isinstance(layer, MaxPooling2D):
            result.push_layer(transfer_maxpooling2d_layer(layer))
        elif isinstance(layer, Flatten):
            result.push_layer(FlattenLayer())
        else:
            print("Warning: skipping layer: %s." % layer)
    return result
