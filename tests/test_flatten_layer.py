#################################################################
# simple_neural_nets
# DNN implementation only relying on a Python base distribution.
# 2024/08/12
# (C) Andreas Gaiser (doraeneko@github)
# Tests for Flatten layer class
#################################################################

import pytest
from .common import *
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten
from ..simple_net.flatten_layer import FlattenLayer
from ..keras2simple_net import transfer_batch_size_1_np_tensor_to_tensor3d, extend_2d_np_tensor_to_4d


@pytest.mark.parametrize(
    "x,y,channels",
    [
        (30, 30, 22),
        (5, 7, 3),
        (5, 17, 1),
        (1, 2, 1),
    ],
)
def test_maxpooling2d_layer(x, y, channels):
    np.random.seed(42)
    tf.random.set_seed(42)
    # neural network with one conv2d layer
    keras_model = Sequential(
        [
            Flatten(
                input_shape=(x, y, channels)
            )
        ]
    )
    keras_model.compile()
    simple_net_instance = FlattenLayer()
    # Test random example value
    keras_input = np.random.rand(1, x, y, channels)
    keras_output = extend_2d_np_tensor_to_4d(keras_model.predict(keras_input))
    simple_net_input = transfer_batch_size_1_np_tensor_to_tensor3d(keras_input)
    simple_net_output = simple_net_instance.output(simple_net_input)
    compare_outputs(keras_output, simple_net_output)
