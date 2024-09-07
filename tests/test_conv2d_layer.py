#################################################################
# simple_neural_nets
# DNN implementation only relying on a Python base distribution.
# 2024/08/12
# (C) Andreas Gaiser (doraeneko@github)
# Tests for Conv2DLayer class
#################################################################

import pytest
from .common import *
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from ..keras2simple_net import (
    transfer_conv2d_layer,
    transfer_batch_size_1_np_tensor_to_tensor3d,
)



@pytest.mark.parametrize(
    "kernel_x, kernel_y, input_x, input_y, channels",
    [
        (3, 3, 10, 10, 4),
        (2, 2, 10, 10, 2),
        (2, 3, 10, 10, 2),
        (3, 2, 5, 7, 2),
        (3, 3, 5, 7, 12),
        (3, 3, 5, 7, 1),
        (4, 4, 100, 100, 5),
        (5, 5, 1, 1, 5),
        (2, 2, 50, 50, 100),
        (3, 3, 50, 50, 100),
        (3, 3, 30, 30, 512),
    ],
)
def test_conv2d_layer_1(kernel_x, kernel_y, input_x, input_y, channels):
    np.random.seed(42)
    tf.random.set_seed(42)
    # neural network with one conv2d layer
    keras_model = Sequential(
        [
            Conv2D(
                filters=2,
                kernel_size=(kernel_x, kernel_y),
                activation="relu",
                padding="same",
                input_shape=(input_x, input_y, channels),
                name="conv_layer",
            )
        ]
    )
    keras_model.compile(optimizer="adam", loss="mean_squared_error")
    simple_net_instance = transfer_conv2d_layer(keras_model.layers[0])
    # Test random example value
    keras_input = np.random.rand(1, input_x, input_y, channels)
    keras_output = keras_model.predict(keras_input)
    simple_net_input = transfer_batch_size_1_np_tensor_to_tensor3d(keras_input)
    simple_net_output = simple_net_instance.output(simple_net_input)
    compare_outputs(keras_output, simple_net_output)
