#################################################################
# simple_neural_nets
# DNN implementation only relying on a Python base distribution.
# 2024/08/12
# (C) Andreas Gaiser (doraeneko@github)
# Tests for MaxPooling2D layer class
#################################################################

import pytest
from .common import *
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import MaxPooling2D
from ..simple_net.maxpooling2d_layer import MaxPooling2DLayer
from ..keras2simple_net import (
    transfer_batch_size_1_np_tensor_to_tensor3d,
    transfer_maxpooling2d_layer,
)


@pytest.mark.parametrize(
    "kernel_x, kernel_y, strides_x, strides_y, input_x, input_y, channels",
    [
        # (2,2,2,2,4,4,1),
        # (2, 2, 3, 3, 5, 5, 1),
        (10, 10, 3, 3, 50, 50, 1),
        (2, 2, 2, 2, 50, 50, 1),
        (2, 2, 2, 2, 50, 50, 2),
        (2, 2, 2, 2, 50, 150, 10),
        # TODO: results differ when kernel and strides are not the same !!!
    ],
)
def test_maxpooling2d_layer(
    kernel_x, kernel_y, strides_x, strides_y, input_x, input_y, channels
):
    np.random.seed(42)
    tf.random.set_seed(42)
    # neural network with one conv2d layer
    keras_model = Sequential(
        [
            MaxPooling2D(
                pool_size=(kernel_x, kernel_y),
                strides=(strides_x, strides_y),
                padding="same",
                input_shape=(input_x, input_y, channels),
                name="max_layer",
            )
        ]
    )
    keras_model.compile()
    simple_net_instance = transfer_maxpooling2d_layer(keras_model.layers[0])
    # Test random example value
    keras_input = np.random.rand(1, input_x, input_y, channels)
    keras_output = keras_model.predict(keras_input)
    simple_net_input = transfer_batch_size_1_np_tensor_to_tensor3d(keras_input)
    simple_net_output = simple_net_instance.output(simple_net_input)
    compare_outputs(keras_output, simple_net_output)
